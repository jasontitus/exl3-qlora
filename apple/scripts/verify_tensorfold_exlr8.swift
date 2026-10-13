// A deliberately scoped before/after parity harness for TensorFold commit
// 51426b8e668b181d28c0bb6964c1882183e292fd: exlr8_projection.metal.
// Load reviewed, compiled libraries only. Does not validate prompt/CUDA/host paths.
import Foundation
import Metal
import CryptoKit
import Dispatch

func sha(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
func failure(_ text: String) -> NSError { NSError(domain: "EXLR8Qualification", code: 1, userInfo: [NSLocalizedDescriptionKey: text]) }
let argv = CommandLine.arguments
if argv.count != 5 { fputs("usage: verify_tensorfold_exlr8 BEFORE.metallib AFTER.metallib COMPILE_LINK_MANIFEST.json NEW_RECEIPT.json\n", stderr); exit(2) }
let receiptURL = URL(fileURLWithPath: argv[4])
if FileManager.default.fileExists(atPath: receiptURL.path) { fputs("refusing to overwrite receipt\n", stderr); exit(2) }
var report: [String: Any] = [
    "format": "exl3-tensorfold-projection-parity-v1",
    "scope": "Finite randomized projection inputs and exhaustive 65536-state codebook before/after parity, on this GPU only; no model or generated-kernel quality claim.",
    "argv": argv, "started_unix": Date().timeIntervalSince1970,
    "kernel_commit": "51426b8e668b181d28c0bb6964c1882183e292fd",
    "precision": "fp32 input and output; original kernel half codebook arithmetic; bitwise parity required",
    "timing": "not_run", "full_commit_verified": false,
    "excluded_paths": ["prompt tensor-unit kernel", "CUDA", "bfloat output", "rotate/combine host integration", "routed members", "exceptional input values"],
    "cases": [[String: Any]](), "passed": false,
]
func save() throws {
    let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: receiptURL, options: .atomic)
}
var rng: UInt64 = 0x51426B8E
func randomWord() -> UInt32 {
    rng = rng &* 6364136223846793005 &+ 1442695040888963407
    return UInt32(truncatingIfNeeded: rng >> 32)
}
func floats(_ b: MTLBuffer, _ count: Int) -> [Float] { Array(UnsafeBufferPointer(start: b.contents().bindMemory(to: Float.self, capacity: count), count: count)) }
do {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw failure("No Metal device") }
    report["device"] = device.name
    report["registry_id"] = String(device.registryID)
    report["os"] = ProcessInfo.processInfo.operatingSystemVersionString
    report["seed"] = String(rng)
    let manifestData = try Data(contentsOf: URL(fileURLWithPath: argv[3]))
    guard let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any],
          manifest["format"] as? String == "exl3-tensorfold-compile-link-v1",
          manifest["commit"] as? String == "51426b8e668b181d28c0bb6964c1882183e292fd",
          let bindings = manifest["libraries"] as? [String: [String: Any]] else { throw failure("Invalid compile/link manifest") }
    let expectedSources = [
        "before": "5f6378c6e2c7a6b74bc9c6e6283a02a2b036693898f1227365e40437927177ba",
        "after": "216dc0cadf64b313283981b107288de42fb551237a1b4390585fc90b101c97de"
    ]
    report["compile_link_manifest_sha256"] = sha(manifestData)
    let executableSHA = sha(try Data(contentsOf: URL(fileURLWithPath: argv[0])))
    guard manifest["harness_executable_sha256"] as? String == executableSHA else { throw failure("Harness executable binding failed") }
    report["harness_executable_sha256"] = executableSHA
    var identities: [[String: Any]] = []
    var libraries: [MTLLibrary] = []
    for (index, label) in ["before", "after"].enumerated() {
        let url = URL(fileURLWithPath: argv[index+1])
        let data = try Data(contentsOf: url)
        guard let binding = bindings[label],
              binding["source_sha256"] as? String == expectedSources[label],
              binding["library_sha256"] as? String == sha(data),
              binding["link_exit_code"] as? Int == 0 else { throw failure("Compile/library binding failed for \(label)") }
        identities.append(["path": url.path, "sha256": sha(data), "bytes": data.count, "source_sha256": expectedSources[label]!])
        let dispatchData = data.withUnsafeBytes { DispatchData(bytes: $0) }
        libraries.append(try device.makeLibrary(data: dispatchData))
    }
    guard identities[0]["sha256"] as? String != identities[1]["sha256"] as? String else { throw failure("Before and after must be distinct bound libraries") }
    report["libraries"] = identities
    func buffer<T>(_ values: [T]) throws -> MTLBuffer {
        guard !values.isEmpty, let b = values.withUnsafeBytes({ raw in device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared) }) else { throw failure("Buffer allocation") }
        return b
    }
    func pipeline(_ library: MTLLibrary, _ name: String) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else { throw failure("Missing kernel \(name)") }
        let state = try device.makeComputePipelineState(function: function)
        guard state.threadExecutionWidth == 32, state.maxTotalThreadsPerThreadgroup >= 256 else { throw failure("Kernel requires SIMD width32 and at least256 threads") }
        return state
    }
    func dispatch(_ state: MTLComputePipelineState, _ buffers: [MTLBuffer], groups: Int, threads: Int) throws {
        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else { throw failure("Command allocation") }
        encoder.setComputePipelineState(state)
        for (index, b) in buffers.enumerated() { encoder.setBuffer(b, offset: 0, index: index) }
        encoder.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        guard command.status == .completed, command.error == nil else { throw failure("GPU failure: \(String(describing: command.error))") }
    }
    func compare(_ a: MTLBuffer, _ b: MTLBuffer, _ count: Int) -> [String: Any] {
        let x = floats(a, count), y = floats(b, count)
        var mismatches = 0, nonfinite = 0
        var maxAbsolute: Float = 0
        for (u, v) in zip(x, y) {
            if u.bitPattern != v.bitPattern { mismatches += 1 }
            if !u.isFinite || !v.isFinite { nonfinite += 1 } else { maxAbsolute = max(maxAbsolute, abs(u-v)) }
        }
        return ["elements": count, "bit_mismatches": mismatches, "nonfinite": nonfinite,
                "max_abs_error": maxAbsolute, "passed": mismatches == 0 && nonfinite == 0,
                "before_sha256": sha(Data(bytes: a.contents(), count: count*4)),
                "after_sha256": sha(Data(bytes: b.contents(), count: count*4))]
    }
    var weightOutputs: [MTLBuffer] = []
    for library in libraries {
        let out = try buffer([Float](repeating: .nan, count: 65536))
        try dispatch(pipeline(library, "tf_exlr8_weights"), [out], groups: 256, threads: 256)
        weightOutputs.append(out)
    }
    let codebook = compare(weightOutputs[0], weightOutputs[1], 65536)
    report["codebook"] = codebook
    try save()
    guard codebook["passed"] as? Bool == true else { throw failure("Codebook parity failed") }
    var cases = [[String: Any]]()
    for capacity in [1, 2, 4, 8, 16] {
        let pipelines = try libraries.map { try pipeline($0, "tf_exlr8_project_\(capacity)") }
        for bits in [2, 3, 4] {
            for blocks in [1, 2, 3] {
                for rows in Array(Set([capacity, max(1, capacity-1)])).sorted() {
                    for raw in [UInt32(0), UInt32(1)] {
                        let input = try buffer((0..<(rows*blocks*128)).map { _ in Float(Int32(bitPattern: randomWord())) / Float(Int32.max) })
                        let packed = try buffer((0..<(blocks*64*8*bits)).map { _ in randomWord() })
                        let bases = try buffer((0..<blocks).map { UInt64($0*64*32*bits) })
                        // Half bit patterns .5,1,2 repeat; scale each input block,
                        // each output channel and final block. Signs alternate.
                        let halfBits = try buffer((0..<(blocks+129)).map { [UInt16(0x3800),UInt16(0x3c00),UInt16(0x4000)][$0%3] })
                        let signs = try buffer((0..<128).map { Int8($0%2 == 0 ? 1 : -1) })
                        let job = try buffer([UInt32(bits),UInt32(blocks),0,0,UInt32(blocks),0,UInt32(blocks+128),0,0,raw,0,0])
                        let args = try buffer([UInt32(rows),UInt32(blocks*128),128,1])
                        let members = try buffer([UInt32(0)])
                        var outputs: [MTLBuffer] = []
                        for p in pipelines {
                            let out = try buffer([Float](repeating: .nan, count: rows*128+32))
                            try dispatch(p, [input,packed,bases,halfBits,signs,job,args,out,members], groups: 1, threads: 256)
                            outputs.append(out)
                        }
                        var row = compare(outputs[0], outputs[1], rows*128)
                        let guarded = outputs.allSatisfy { floats($0, rows*128+32).suffix(32).allSatisfy { $0.bitPattern == Float.nan.bitPattern } }
                        row["guard_unchanged"] = guarded
                        row["guard_scope"] = "32-float trailing output canary, exact bits"
                        row["passed"] = row["passed"] as? Bool == true && guarded
                        row["capacity"] = capacity; row["rows"] = rows; row["bits"] = bits; row["blocks"] = blocks; row["raw"] = raw
                        row["input_sha256"] = sha(Data(bytes: input.contents(), count: input.length))
                        row["packed_sha256"] = sha(Data(bytes: packed.contents(), count: packed.length))
                        cases.append(row); report["cases"] = cases
                        try save()
                        guard row["passed"] as? Bool == true else { throw failure("Projection parity failed") }
                    }
                }
            }
        }
    }
    report["case_count"] = cases.count
    report["passed"] = true
    report["completed_unix"] = Date().timeIntervalSince1970
    try save()
    print("PASS: exhaustive codebook and \(cases.count) projection cases; no timing or whole-commit claim.")
} catch {
    report["passed"] = false
    report["error"] = String(describing: error)
    report["completed_unix"] = Date().timeIntervalSince1970
    try? save()
    fputs("FAIL: \(error)\n", stderr)
    exit(1)
}
