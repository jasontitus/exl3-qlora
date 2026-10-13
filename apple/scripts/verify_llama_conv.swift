// Fixed, scoped parity test for llama.cpp 560445bf34c87356ad0f8d80fb03ec5488850b65.
// No performance measurement. Run only after unrelated GPU work has stopped.
import Foundation
import Metal
import CryptoKit
import Dispatch
import Darwin

func sha(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
func fail(_ text: String) -> NSError { NSError(domain: "LlamaConvParity", code: 1, userInfo: [NSLocalizedDescriptionKey: text]) }
func dictionary(_ data: Data) throws -> [String: Any] {
    guard let result = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw fail("Expected JSON object") }
    return result
}
func checkedFile(_ entry: [String: Any]) throws -> Data {
    guard let path = entry["path"] as? String, let expected = entry["sha256"] as? String else { throw fail("Missing file binding") }
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    guard sha(data) == expected else { throw fail("File binding failed: \(path)") }
    return data
}
struct Shape {
    let ic: Int, il: Int, k: Int, stride: Int, oc: Int, extra: Int, columnStride: Int, rowPadding: Int
    var ol: Int { (il - 1) * stride + k + extra }
    var rowWords: Int { ol * columnStride + rowPadding }
    var json: [String: Any] { ["input_channels": ic, "input_length": il, "kernel_size": k, "stride": stride, "output_channels": oc, "output_length": ol, "extra_right_outputs": extra, "output_column_stride_words": columnStride, "output_row_padding_words": rowPadding] }
}
// Selected before execution: singletons, overlapping kernels, stride>K holes,
// beyond-support outputs, odd extents and the source-reported codec geometry.
let shapes = [
    Shape(ic: 1, il: 1, k: 1, stride: 1, oc: 1, extra: 3, columnStride: 1, rowPadding: 3),
    Shape(ic: 2, il: 1, k: 7, stride: 3, oc: 3, extra: 4, columnStride: 2, rowPadding: 5),
    Shape(ic: 3, il: 7, k: 1, stride: 4, oc: 2, extra: 4, columnStride: 1, rowPadding: 7),
    Shape(ic: 5, il: 9, k: 3, stride: 5, oc: 3, extra: 6, columnStride: 2, rowPadding: 3),
    Shape(ic: 3, il: 17, k: 7, stride: 1, oc: 4, extra: 2, columnStride: 1, rowPadding: 5),
    Shape(ic: 7, il: 33, k: 10, stride: 5, oc: 5, extra: 7, columnStride: 2, rowPadding: 7),
    Shape(ic: 5, il: 65, k: 17, stride: 3, oc: 3, extra: 5, columnStride: 1, rowPadding: 3),
    Shape(ic: 8, il: 320, k: 10, stride: 5, oc: 8, extra: 5, columnStride: 1, rowPadding: 5),
]
struct Args { var ic: Int32; var il: Int32; var k: Int32; var stride: Int32; var nb0: UInt64; var nb1: UInt64 }
let argv = CommandLine.arguments
if !(argv.count == 3 || (argv.count == 4 && argv[3] == "--validate-only")) {
    fputs("usage: verify_llama_conv COMPILE_LINK_MANIFEST.json NEW_RECEIPT.json [--validate-only]\n", stderr); exit(2)
}
let output = URL(fileURLWithPath: argv[2]); let fd = Darwin.open(output.path, O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR)
guard fd >= 0 else { fputs("Cannot exclusively create new receipt\n", stderr); exit(2) }
let file = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
var report: [String: Any] = [
    "format": "exl3-llama-conv-parity-v1", "argv": argv, "started_unix": Date().timeIntervalSince1970,
    "scope": "Exact reviewed isolated conv_transpose_1d kernel pair, deterministic finite dyadic inputs and fixed shapes on this GPU; not whole-commit or generated-kernel quality verification.",
    "commit": "560445bf34c87356ad0f8d80fb03ec5488850b65", "parent": "2eb3e6b2428c035dd24f996c7dfc48654dc19a6b",
    "shape_plan": shapes.map { $0.json }, "weight_types": ["float32", "float16"], "case_count_expected": shapes.count * 2,
    "input_generator": "x[i]=((13*i)%23-11)/16; w[i]=((17*i)%31-15)/32; exactly representable in FP16/FP32; finite dyadic CPU reference is exact for these bounded sums",
    "dispatch": "One thread per output; threadgroups=(output_length, output_channels, 1), threads_per_threadgroup=(1,1,1)",
    "numerical_parity": "not_run", "performance": "not_run", "full_commit_verified": false,
    "limitations": ["No timing or speed claim", "No surrounding ggml runtime or full-model validation", "No NaN/Inf, overflow, zero/negative strides or arbitrary-shape claim", "Fixed dyadic values test geometry and bitwise preservation, not all rounding-sensitive distributions", "Compiler validation flags may differ from the upstream build"],
    "cases": [[String: Any]](), "passed": false,
]
func save() throws {
    let bytes = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    try file.seek(toOffset: 0); try file.truncate(atOffset: 0); try file.write(contentsOf: bytes); try file.synchronize()
}
do {
    try save()
    let manifestData = try Data(contentsOf: URL(fileURLWithPath: argv[1])); let manifest = try dictionary(manifestData)
    guard manifest["format"] as? String == "exl3-llama-conv-compile-link-v1",
          manifest["commit"] as? String == report["commit"] as? String,
          manifest["parent"] as? String == report["parent"] as? String,
          let files = manifest["files"] as? [String: [String: Any]],
          let libs = manifest["libraries"] as? [String: [String: Any]],
          let executable = files["harness_binary"], let source = files["harness_source"],
          let compileBinding = files["compile_receipt"], let compositionBinding = files["composition"] else { throw fail("Invalid manifest") }
    let executableData = try checkedFile(executable)
    guard executableData == (try Data(contentsOf: URL(fileURLWithPath: argv[0]))) else { throw fail("Running executable mismatch") }
    _ = try checkedFile(source)
    let compiled = try dictionary(checkedFile(compileBinding)); let composition = try dictionary(checkedFile(compositionBinding))
    guard compiled["compiled"] as? Bool == true,
          let compiles = compiled["compiles"] as? [String: [String: Any]],
          let sources = compiled["sources"] as? [String: [String: Any]],
          let sides = composition["sides"] as? [String: [String: Any]],
          composition["commit"] as? String == report["commit"] as? String,
          composition["parent"] as? String == report["parent"] as? String else { throw fail("Compile/composition binding failed") }
    var libraryData = [Data](); var libraryIdentities = [[String: Any]]()
    let reviewedSourceSHA = [
        "before": "cdd40eec64876921c7c26b6a920d9a53218ad5f29cdda5ce2154ec94127e118e",
        "after": "d7d32a6f9e7e8a2dd4491f08a5d84aa04d295ec9cec05054c20dc46d888d2575",
    ]
    for label in ["before", "after"] {
        guard let binding = libs[label], let airEntry = binding["air"] as? [String: Any], let sourceEntry = binding["source"] as? [String: Any],
              let side = sides[label], let pieces = side["pieces"] as? [[String: Any]],
              binding["link_exit_code"] as? Int == 0 else { throw fail("Missing ordered library binding") }
        let data = try checkedFile(binding); let air = try checkedFile(airEntry); let metal = try checkedFile(sourceEntry)
        guard sha(metal) == reviewedSourceSHA[label],
              compiles[label]?["exit_code"] as? Int == 0,
              (compiles[label]?["error"] == nil || compiles[label]?["error"] is NSNull),
              compiles[label]?["air_sha256"] as? String == sha(air), compiles[label]?["passed"] as? Bool == true,
              sources[label]?["sha256"] as? String == sha(metal), side["sha256"] as? String == sha(metal) else { throw fail("AIR/source compile mismatch") }
        var composed = Data()
        for (index, piece) in pieces.enumerated() {
            guard let path = piece["path"] as? String, let fullSHA = piece["full_file_sha256"] as? String,
                  let slice = piece["binding"] as? [String: Any], let start = slice["start"] as? Int, let end = slice["end"] as? Int,
                  let digest = slice["sha256"] as? String else { throw fail("Invalid source slice") }
            let raw = try Data(contentsOf: URL(fileURLWithPath: path))
            guard sha(raw) == fullSHA, start >= 0, end >= start, end <= raw.count else { throw fail("Pinned full source mismatch") }
            let unit = raw.subdata(in: start..<end); guard sha(unit) == digest else { throw fail("Reviewed unit mismatch") }
            if index > 0 { composed.append(Data("\n\n".utf8)) }; composed.append(unit)
        }
        composed.append(10); guard composed == metal else { throw fail("Exact composition mismatch") }
        libraryData.append(data); libraryIdentities.append(["label": label, "sha256": sha(data), "source_sha256": sha(metal), "air_sha256": sha(air)])
    }
    guard sha(libraryData[0]) != sha(libraryData[1]), MemoryLayout<Args>.stride == 32,
          MemoryLayout<Args>.offset(of: \.nb0) == 16, MemoryLayout<Args>.offset(of: \.nb1) == 24 else { throw fail("Library identity or argument ABI mismatch") }
    report["manifest_sha256"] = sha(manifestData); report["harness_sha256"] = sha(executableData); report["libraries"] = libraryIdentities; report["bindings_verified"] = true
    if argv.count == 4 {
        report["status"] = "validated_without_gpu_execution"; report["finished_unix"] = Date().timeIntervalSince1970; try save(); print("Bindings and case plan verified; GPU not executed"); exit(0)
    }
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw fail("Metal unavailable") }
    report["device"] = device.name; report["registry_id"] = String(device.registryID); report["os"] = ProcessInfo.processInfo.operatingSystemVersionString
    let libraries = try libraryData.map { data in try device.makeLibrary(data: data.withUnsafeBytes { DispatchData(bytes: $0) }) }
    func buffer<T>(_ values: [T]) throws -> MTLBuffer {
        guard let result = values.withUnsafeBytes({ raw in device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared) }) else { throw fail("Buffer allocation failed") }; return result
    }
    let canary: UInt32 = 0x7fc01234, prefixWords = 16, tailWords = 32
    var results = [[String: Any]]()
    for halfWeights in [false, true] {
        let functionName = halfWeights ? "kernel_conv_transpose_1d_f16_f32" : "kernel_conv_transpose_1d_f32_f32"
        let pipelines = try libraries.map { library -> MTLComputePipelineState in
            guard let function = library.makeFunction(name: functionName) else { throw fail("Missing function \(functionName)") }
            return try device.makeComputePipelineState(function: function)
        }
        for (shapeIndex, s) in shapes.enumerated() {
            let inputs = (0..<(s.ic*s.il)).map { Float(($0*13)%23-11)/16 }
            let weights = (0..<(s.ic*s.oc*s.k)).map { Float(($0*17)%31-15)/32 }
            let x = try buffer(inputs); let w = try halfWeights ? buffer(weights.map { Float16($0) }) : buffer(weights)
            let xOriginal = Data(bytes: x.contents(), count: x.length); let wOriginal = Data(bytes: w.contents(), count: w.length)
            let totalWords = prefixWords + s.rowWords*s.oc + tailWords
            var active = Set<Int>(); var expected = [UInt32: Float]()
            for oc in 0..<s.oc { for j in 0..<s.ol {
                let offset = prefixWords + oc*s.rowWords + j*s.columnStride; active.insert(offset)
                var acc: Float = 0
                for c in 0..<s.ic { for i in 0..<s.il { let k = j-i*s.stride
                    if k >= 0 && k < s.k { acc += weights[c*s.oc*s.k+oc*s.k+k] * inputs[c*s.il+i] }
                }}
                expected[UInt32(offset)] = acc
            }}
            var outputs = [[UInt32]](); var inputMutations = 0
            for p in pipelines {
                let out = try buffer([UInt32](repeating: canary, count: totalWords))
                var args = Args(ic: Int32(s.ic), il: Int32(s.il), k: Int32(s.k), stride: Int32(s.stride), nb0: UInt64(s.columnStride*4), nb1: UInt64(s.rowWords*4))
                guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else { throw fail("Command allocation failed") }
                encoder.setComputePipelineState(p); encoder.setBytes(&args, length: MemoryLayout<Args>.stride, index: 0)
                encoder.setBuffer(w, offset: 0, index: 1); encoder.setBuffer(x, offset: 0, index: 2); encoder.setBuffer(out, offset: prefixWords*4, index: 3)
                encoder.dispatchThreadgroups(MTLSize(width: s.ol, height: s.oc, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                guard command.status == .completed, command.error == nil else { throw fail("GPU command failed: \(String(describing: command.error))") }
                outputs.append(Array(UnsafeBufferPointer(start: out.contents().bindMemory(to: UInt32.self, capacity: totalWords), count: totalWords)))
                if Data(bytes: x.contents(), count: x.length) != xOriginal || Data(bytes: w.contents(), count: w.length) != wOriginal { inputMutations += 1 }
            }
            var differences = 0, nonfinite = 0, canaryChanges = 0, oracleDifferences = 0
            for index in 0..<totalWords {
                if active.contains(index) {
                    if outputs[0][index] != outputs[1][index] { differences += 1 }
                    for values in outputs {
                        let f = Float(bitPattern: values[index]); if !f.isFinite { nonfinite += 1 }
                        if f.bitPattern != expected[UInt32(index)]!.bitPattern { oracleDifferences += 1 }
                    }
                } else { for values in outputs { if values[index] != canary { canaryChanges += 1 } } }
            }
            let passed = differences == 0 && nonfinite == 0 && canaryChanges == 0 && oracleDifferences == 0 && inputMutations == 0
            let raw = outputs.map { words in words.withUnsafeBytes { Data($0) } }
            results.append(["shape_index": shapeIndex, "shape": s.json, "weights": halfWeights ? "float16" : "float32", "output_elements": active.count, "canary_words_per_variant": totalWords-active.count, "bit_mismatches": differences, "nonfinite_values": nonfinite, "cpu_oracle_bit_mismatches": oracleDifferences, "canary_mutations": canaryChanges, "input_mutations": inputMutations, "before_output_buffer_sha256": sha(raw[0]), "after_output_buffer_sha256": sha(raw[1]), "input_sha256": sha(xOriginal), "weights_sha256": sha(wOriginal), "passed": passed])
            report["cases"] = results; try save()
            guard passed else { throw fail("Parity/canary/oracle check failed at case \(shapeIndex), half=\(halfWeights)") }
        }
    }
    guard results.count == shapes.count*2 else { throw fail("Incomplete case count") }
    report["passed"] = true; report["numerical_parity"] = "passed_fixed_cases"; report["finished_unix"] = Date().timeIntervalSince1970; try save(); print("All16 fixed convolution cases passed; no speed claim")
} catch {
    report["passed"] = false; report["numerical_parity"] = "failed_or_incomplete"
    report["error"] = String(describing: error); report["finished_unix"] = Date().timeIntervalSince1970
    try? save(); fputs("\(error)\n", stderr); exit(1)
}
