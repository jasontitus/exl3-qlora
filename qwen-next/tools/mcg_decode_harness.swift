// Standalone synthetic decoder qualification. --validate-only never creates a GPU device.
import Foundation
import CoreFoundation
import CryptoKit
import Metal

struct Rejected: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}
func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw Rejected(message) }
}
func sha(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
func boundedFile(_ path: String, _ limit: Int) throws -> Data {
    let before = try FileManager.default.attributesOfItem(atPath: path)
    try require(before[.type] as? FileAttributeType == .typeRegular, "Expected regular file: \(path)")
    let size = (before[.size] as? NSNumber)?.intValue ?? -1
    try require(size > 0 && size <= limit, "File exceeds bounded size: \(path)")
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let after = try FileManager.default.attributesOfItem(atPath: path)
    for key: FileAttributeKey in [.size, .modificationDate, .systemFileNumber, .systemNumber] {
        try require(String(describing: before[key]) == String(describing: after[key]), "File changed during read")
    }
    try require(data.count == size, "File size changed during read")
    return data
}
func integer(_ value: Any?, _ name: String, min: Int, max: Int) throws -> Int {
    guard let number = value as? NSNumber else { throw Rejected("Missing integer \(name)") }
    try require(CFGetTypeID(number) != CFBooleanGetTypeID(), "Boolean is not integer \(name)")
    // JSONSerialization sometimes canonicalizes 1.0 to an integer NSNumber. The contract
    // is an exact integer value, not lexical JSON number spelling.
    let d = number.doubleValue
    try require(d.isFinite && d.rounded() == d && d >= Double(min) && d <= Double(max), "Invalid integer \(name)")
    return number.intValue
}
func words(_ value: Any?, count: Int) throws -> [UInt16] {
    guard let list = value as? [Any] else { throw Rejected("Expected word array") }
    try require(list.count == count, "Word count does not match geometry")
    return try list.map { UInt16(try integer($0, "word", min: 0, max: 65535)) }
}
func bytes(_ words: [UInt16]) -> Data {
    var result = Data(capacity: words.count * 2)
    for word in words { result.append(UInt8(word & 255)); result.append(UInt8(word >> 8)) }
    return result
}
struct Case {
    let id: String
    let k, n, start, width, halfwords, window: Int
    let packed, expected: [UInt16]
    init(_ object: [String: Any]) throws {
        try require(Set(object.keys) == Set(["case_id", "k", "n", "start", "width", "halfwords", "window", "packed", "expected_fp16_bits", "packed_sha256", "expected_sha256"]), "Unknown or missing case fields")
        guard let id = object["case_id"] as? String else { throw Rejected("Missing case ID") }
        try require(!id.isEmpty && id.utf8.count <= 100, "Invalid case ID")
        self.id = id
        k = try integer(object["k"], "k", min: 16, max: 256)
        n = try integer(object["n"], "n", min: 16, max: 256)
        start = try integer(object["start"], "start", min: 0, max: 240)
        width = try integer(object["width"], "width", min: 16, max: 256)
        halfwords = try integer(object["halfwords"], "halfwords", min: 16, max: 128)
        window = try integer(object["window"], "window", min: 8, max: 16)
        try require(k % 16 == 0 && n % 16 == 0 && start % 16 == 0 && width % 16 == 0 && start + width <= n && halfwords % 2 == 0, "Unsupported aligned geometry or rate")
        packed = try words(object["packed"], count: (k / 16) * (n / 16) * halfwords)
        expected = try words(object["expected_fp16_bits"], count: k * width)
        try require(expected.allSatisfy { ($0 & 0x7c00) != 0x7c00 }, "Nonfinite expected FP16 value")
        try require(object["packed_sha256"] as? String == sha(bytes(packed)), "Packed payload hash mismatch")
        try require(object["expected_sha256"] as? String == sha(bytes(expected)), "Expected payload hash mismatch")
    }
}
func runCase(_ c: Case, _ device: MTLDevice, _ queue: MTLCommandQueue, _ pipeline: MTLComputePipelineState) throws -> [String: Any] {
    let guardWords = 32, sentinel = UInt16(0x7e55)
    let inputWords = [UInt16](repeating: sentinel, count: guardWords) + c.packed + [UInt16](repeating: sentinel, count: guardWords)
    let outputWords = [UInt16](repeating: sentinel, count: c.expected.count + 2 * guardWords)
    let inputData = bytes(inputWords), outputData = bytes(outputWords)
    let input = inputData.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
    let output = outputData.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
    guard let input, let output, let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else { throw Rejected("Metal allocation/encoder failed") }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: guardWords * 2, index: 0)
    encoder.setBuffer(output, offset: guardWords * 2, index: 1)
    for (i, value) in [c.k, c.n, c.start, c.width, c.halfwords, c.window].enumerated() {
        var parameter = UInt32(value)
        encoder.setBytes(&parameter, length: MemoryLayout<UInt32>.size, index: i + 2)
    }
    // Deliberately over-dispatch to test gid >= K*width without writing suffix guards.
    encoder.dispatchThreads(MTLSize(width: c.expected.count + 37, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(64, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    try require(command.status == .completed, "Metal command failed: \(String(describing: command.error))")
    let actual = output.contents().bindMemory(to: UInt16.self, capacity: outputWords.count)
    var mismatches = 0, first: [[String: Int]] = []
    for i in 0..<c.expected.count where actual[i + guardWords] != c.expected[i] {
        mismatches += 1
        if first.count < 8 { first.append(["index": i, "expected_bits": Int(c.expected[i]), "actual_bits": Int(actual[i + guardWords])]) }
    }
    let guardsOK = (0..<guardWords).allSatisfy { actual[$0] == sentinel && actual[guardWords + c.expected.count + $0] == sentinel }
    let inputOK = Data(bytes: input.contents(), count: input.length) == inputData
    let actualBytes = Data(bytes: output.contents().advanced(by: guardWords * 2), count: c.expected.count * 2)
    return ["case_id": c.id, "k": c.k, "n": c.n, "start": c.start, "width": c.width, "halfwords": c.halfwords, "window": c.window,
            "compared_values": c.expected.count, "mismatches": mismatches, "first_mismatches": first,
            "output_guards_unchanged": guardsOK, "input_and_guards_unchanged": inputOK,
            "actual_sha256": sha(actualBytes), "passed": mismatches == 0 && guardsOK && inputOK]
}
func main() throws -> Int32 {
    var args = Array(CommandLine.arguments.dropFirst()), options: [String: String] = [:]
    var mode: String? = nil
    while !args.isEmpty {
        let key = args.removeFirst()
        if key == "--validate-only" || key == "--dispatch" {
            try require(mode == nil, "Choose exactly one mode")
            mode = key
        } else {
            try require(["--fixtures", "--kernel", "--receipt"].contains(key) && options[key] == nil && !args.isEmpty, "Unknown, duplicate or incomplete argument \(key)")
            options[key] = args.removeFirst()
        }
    }
    guard let mode, let fixturePath = options["--fixtures"], let kernelPath = options["--kernel"], let receipt = options["--receipt"] else { throw Rejected("Require --validate-only|--dispatch --fixtures FILE --kernel FILE --receipt NEW_FILE") }
    try require(!FileManager.default.fileExists(atPath: receipt), "Receipt exists; refusing overwrite")
    let fixtureData = try boundedFile(fixturePath, 8 << 20), kernelData = try boundedFile(kernelPath, 64 << 10)
    guard let suite = try JSONSerialization.jsonObject(with: fixtureData) as? [String: Any], let rawCases = suite["cases"] as? [[String: Any]] else { throw Rejected("Invalid fixture JSON") }
    try require(Set(suite.keys) == Set(["schema", "scope", "comparison", "kernel_sha256", "cpu_reference_sha256", "upstream_sha256", "generator_sha256", "limitations", "cases"]), "Unknown or missing suite fields")
    try require(suite["schema"] as? String == "qwen-mcg-metal-synthetic-v1" && suite["scope"] as? String == "synthetic-inner-mcg-only" && suite["comparison"] as? String == "exact-fp16-bits", "Unsupported fixture contract")
    try require(suite["kernel_sha256"] as? String == sha(kernelData), "Kernel identity mismatch")
    try require(suite["cpu_reference_sha256"] as? String == "36827164c88dd6c25dfde67077abe3ad5fb45067f85940a64035fe79dbf29493" && suite["upstream_sha256"] as? String == "33d37858ec6647c60fb975e5c011b1354aa4453e9258a0e79a1f781d22a5e0a5", "Unreviewed reference identity")
    guard let generator = suite["generator_sha256"] as? String else { throw Rejected("Missing generator identity") }
    try require(generator.count == 64 && generator.allSatisfy { "0123456789abcdef".contains($0) }, "Invalid generator identity")
    try require(!rawCases.isEmpty && rawCases.count <= 128, "Fixture case limit exceeded")
    let cases = try rawCases.map { try Case($0) }
    try require(Set(cases.map { $0.id }).count == cases.count, "Duplicate case ID")
    try require(cases.reduce(0) { $0 + $1.expected.count } <= (1 << 20), "Aggregate output limit exceeded")
    var report: [String: Any] = ["schema": "qwen-mcg-metal-harness-v1", "mode": mode, "fixture_sha256": sha(fixtureData), "kernel_sha256": sha(kernelData), "cpu_reference_sha256": suite["cpu_reference_sha256"]!, "generator_sha256": generator, "case_count": cases.count, "validated_values": cases.reduce(0) { $0 + $1.expected.count }, "validation_passed": true, "gpu_executed": false, "numerical_parity_qualified": false, "comparison": "exact-fp16-bits", "tolerance": 0, "argv": CommandLine.arguments, "process_id": ProcessInfo.processInfo.processIdentifier, "os": ProcessInfo.processInfo.operatingSystemVersionString, "limitations": ["Synthetic inner MCG decoder only", "No real checkpoint/converter-window14 qualification, public transform, serving, matmul, VJP, model or training claim", "Fixture hashes check consistency, not authenticity of independently supplied expected answers"]]
    let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL.path
    report["executable_sha256"] = sha(try boundedFile(executable, 64 << 20))
    var exitCode: Int32 = 0
    if mode == "--dispatch" {
        guard let source = String(data: kernelData, encoding: .utf8), let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { throw Rejected("Metal unavailable or source not UTF8") }
        let compile = MTLCompileOptions()
        compile.fastMathEnabled = false
        compile.languageVersion = .version3_1
        let library = try device.makeLibrary(source: source, options: compile)
        guard let function = library.makeFunction(name: "qwen_mcg_decode") else { throw Rejected("Decoder function missing") }
        let pipeline = try device.makeComputePipelineState(function: function)
        var results: [[String: Any]] = []
        for c in cases { results.append(try runCase(c, device, queue, pipeline)) }
        let passed = results.allSatisfy { $0["passed"] as? Bool == true }
        report["gpu_executed"] = true
        report["numerical_parity_qualified"] = passed
        report["qualification_scope"] = "Only the exact synthetic cases and hardware recorded here"
        report["device"] = device.name
        report["device_registry_id"] = String(device.registryID)
        report["fast_math_enabled"] = false
        report["metal_language"] = "3.1"
        report["results"] = results
        if !passed { exitCode = 2 }
    }
    report["finished_utc"] = ISO8601DateFormatter().string(from: Date())
    let encoded = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys, .prettyPrinted])
    try encoded.write(to: URL(fileURLWithPath: receipt), options: .withoutOverwriting)
    print("\(mode): validated \(cases.count) cases; GPU executed: \(mode == "--dispatch"); receipt: \(receipt)")
    return exitCode
}
do { exit(try main()) } catch {
    FileHandle.standardError.write(Data("Rejected: \(error)\n".utf8))
    exit(1)
}
