// Bounded correctness of two generated expressions in real held-out Metal kernels.
// This is not an optimization benchmark or a whole-repository build.
import Foundation
import Metal
import CryptoKit
import Dispatch
import Darwin

let pinnedPlanSHA = "786d572c19ebe0e7cd14ea3e060749749f93e12d845c09526d67b3d22de742c1"
func digest(_ d: Data) -> String { SHA256.hash(data:d).map{String(format:"%02x",$0)}.joined() }
func failure(_ s: String) -> NSError { NSError(domain:"ExpressionProbe",code:1,userInfo:[NSLocalizedDescriptionKey:s]) }
func object(_ d: Data) throws -> [String:Any] { guard let o=try JSONSerialization.jsonObject(with:d) as? [String:Any] else {throw failure("Expected object")};return o }
func checked(_ b: [String:Any]) throws -> Data { guard let p=b["path"] as? String,let s=b["sha256"] as? String else {throw failure("Missing binding")};let d=try Data(contentsOf:URL(fileURLWithPath:p));guard digest(d)==s else {throw failure("Changed file: \(p)")};return d }
func binding(_ parent:[String:Any],_ key:String)throws -> [String:Any] {guard let b=parent[key] as? [String:Any] else {throw failure("Missing \(key)")};return b}
func equalJSON(_ a:Any,_ b:Any)throws -> Bool {try JSONSerialization.data(withJSONObject:a,options:[.sortedKeys]) == JSONSerialization.data(withJSONObject:b,options:[.sortedKeys])}

// Pure nonnegative index grammar; no evaluate/NSExpression/dynamic host code.
struct IndexParser {
    var input:[Character];var pos=0;let index:Int64;let count:Int64
    mutating func skip(){while pos<input.count && input[pos]==" "{pos+=1}}
    mutating func take(_ c:Character)->Bool {skip();if pos<input.count && input[pos]==c{pos+=1;return true};return false}
    mutating func bound(_ n:Int64)throws->Int64{guard n>=0 && n<Int64(Int32.max) else{throw failure("Index intermediate exceeds signed32 bound")};return n}
    mutating func atom()throws->Int64 {
        skip();if take("("){let v=try sum();guard take(")") else{throw failure("Unclosed expression")};return v}
        let start=pos
        while pos<input.count && (input[pos].isLetter || input[pos].isNumber || input[pos]=="_"){pos+=1}
        let s=String(input[start..<pos]);if s=="tpig"{return index};if s=="nb"{return count}
        guard !s.isEmpty,s.allSatisfy({$0.isNumber}),let n=Int64(s),n<=1000000 else{throw failure("Invalid index token")};return n
    }
    mutating func product()throws->Int64 {var v=try atom();while true{if take("*"){v=try bound(v * atom())}else if take("%"){let n=try atom();guard n>0 else{throw failure("Zero index divisor")};v=v%n}else{return v}}}
    mutating func sum()throws->Int64 {var v=try product();while take("+"){v=try bound(v + product())};return v}
    mutating func parse()throws->Int64 {guard input.count<=128 else{throw failure("Long index")};let v=try sum();skip();guard pos==input.count else{throw failure("Trailing index syntax")};return v}
}
func validateExpression(_ expression:String,operation:String,cases:[[String:Any]])throws {
    guard !expression.isEmpty,expression.count<=128,!expression.contains("\n"),!expression.contains("\r") else{throw failure("Invalid expression length")}
    if operation=="broadcast_mul" {
        for c in cases {guard let n=c["vectors_per_row"] as? Int,let rows=c["rows"] as? Int,n>0,rows>0 else{throw failure("Invalid case")}
            for i in 0..<n*rows {var parser=IndexParser(input:Array(expression),index:Int64(i),count:Int64(n));let v=try parser.parse();guard v>=0 && v<n else{throw failure("Index outside row")}}
        }
    } else {
        guard expression.range(of:#"^[A-Za-z0-9_+*()., /-]+$"#,options:.regularExpression) != nil,!expression.contains("//"),!expression.contains("/*") else{throw failure("Invalid scalar grammar")}
        let rx=try NSRegularExpression(pattern:"[A-Za-z_][A-Za-z_0-9]*")
        for match in rx.matches(in:expression,range:NSRange(expression.startIndex...,in:expression)) {
            let name=(expression as NSString).substring(with:match.range)
            guard ["sumf","simd_sum","simd_min","simd_max"].contains(name) else{throw failure("Unsupported scalar name")}
        }
    }
}
let argv=CommandLine.arguments
if !(argv.count==3 || (argv.count==4 && argv[3]=="--validate-only")){fputs("usage: verify_expression_probes JOB.json NEW_RECEIPT.json [--validate-only]\n",stderr);exit(2)}
let output=URL(fileURLWithPath:argv[2]);let fd=Darwin.open(output.path,O_WRONLY|O_CREAT|O_EXCL,S_IRUSR|S_IWUSR)
guard fd>=0 else{fputs("Receipt exists or cannot be created\n",stderr);exit(2)}
let file=FileHandle(fileDescriptor:fd,closeOnDealloc:true)
var report:[String:Any]=["format":"heldout-expression-result-v1","argv":argv,"started_unix":Date().timeIntervalSince1970,"passed":false,"gpu_executed":false,"numerical":"not_run","performance":"not_run","scope":"Two predeclared expression completions, exact fixed source composition, bounded finite inputs and independent CPU references. No optimization discovery, whole patch generation, or speed claim.","cases":[[String:Any]]()]
func save()throws{let d=try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]);try file.seek(toOffset:0);try file.truncate(atOffset:0);try file.write(contentsOf:d);try file.synchronize()}
do {
    try save();let jobRaw=try Data(contentsOf:URL(fileURLWithPath:argv[1]));let job=try object(jobRaw)
    guard job["format"] as? String=="heldout-expression-job-v1" else{throw failure("Wrong job format")}
    let planRaw=try checked(binding(job,"plan"));guard digest(planRaw)==pinnedPlanSHA else{throw failure("Wrong predeclared plan")};let plan=try object(planRaw)
    let responseRaw=try checked(binding(job,"responses"));let responses=try object(responseRaw)
    guard responses["plan_sha256"] as? String==pinnedPlanSHA,responses["origin"] as? String==job["origin"] as? String else{throw failure("Response identity mismatch")}
    let executable=try checked(binding(job,"harness_binary"));guard executable==(try Data(contentsOf:URL(fileURLWithPath:argv[0]))) else{throw failure("Wrong executable")}
    _=try checked(binding(job,"harness_source"));_=try checked(binding(job,"builder"))
    guard let probes=job["probes"] as? [[String:Any]],let planned=plan["probes"] as? [[String:Any]],probes.count==2,planned.count==2 else{throw failure("Expected exactly two probes")}
    guard let answers=responses["responses"] as? [String:[String:Any]],Set(answers.keys)==Set(planned.map{$0["id"] as! String}) else{throw failure("Response set differs from plan")}
    var captured=[[String:Data]]();var statuses=[String]()
    for (i,p) in probes.enumerated() {
        let q=planned[i]
        guard p["id"] as? String==q["id"] as? String,p["operation"] as? String==q["operation"] as? String,p["function"] as? String==q["function"] as? String,try equalJSON(p["cases"]!,q["cases"]!) else{throw failure("Probe plan changed")}
        let reference=try binding(p,"reference");guard reference["passed"] as? Bool==true else{throw failure("Reference compile failed")}
        let golden=try checked(binding(q,"golden_source"));let template=String(data:try checked(binding(q,"template")),encoding:.utf8)!
        guard try checked(binding(reference,"source"))==golden else{throw failure("Wrong reference source")}
        _=try checked(binding(reference,"air"));var libs=["reference":try checked(binding(reference,"library"))]
        let status=p["candidate_status"] as? String ?? "missing";statuses.append(status)
        if status=="compiled" {
            let candidate=try binding(p,"candidate");guard candidate["passed"] as? Bool==true,let expression=p["expression"] as? String,let cases=p["cases"] as? [[String:Any]],let operation=p["operation"] as? String else{throw failure("Candidate missing")}
            guard let answer=answers[p["id"] as! String],let rawText=answer["text"] as? String,p["raw_text"] as? String==rawText,expression==rawText.trimmingCharacters(in:.whitespacesAndNewlines) else{throw failure("Candidate differs from supplied response")}
            if responses["origin"] as? String=="model_generation" {
                let generationData=try checked(binding(p,"generation_receipt"));let generation=try object(generationData)
                let originalBinding=try binding(answer,"generation_receipt")
                let expectedPromptIDs = try JSONSerialization.jsonObject(with: checked(binding(q, "prompt_token_ids")))
                guard digest(generationData) == originalBinding["sha256"] as? String,
                      generation["text"] as? String == rawText,
                      generation["max_new_tokens"] as? Int == 8,
                      generation["method"] as? String == "greedy full-prefix recomputation",
                      let ids = generation["generated_ids"] as? [Int],
                      ids.count > 0, ids.count <= 8,
                      let actualPromptIDs = generation["prompt_ids"],
                      try equalJSON(actualPromptIDs, expectedPromptIDs)
                else { throw failure("Generated token/prompt binding failed") }
            } else {
                let expected=responses["origin"] as? String=="source_reference_not_model" ? q["golden"] as? String:q["negative_control"] as? String
                guard rawText==expected else{throw failure("Control differs from plan")}
            }
            try validateExpression(expression,operation:operation,cases:cases)
            let source=try checked(binding(candidate,"source"));guard source==Data(template.replacingOccurrences(of:"__HOLE__",with:expression).utf8) else{throw failure("Generated expression/source mismatch")}
            _=try checked(binding(candidate,"air"));libs["candidate"]=try checked(binding(candidate,"library"))
        }
        captured.append(libs)
    }
    report["job_sha256"]=digest(jobRaw);report["plan_sha256"]=pinnedPlanSHA;report["response_sha256"]=digest(responseRaw);report["origin"]=job["origin"];report["compiler"]=job["compiler"];report["bindings_verified"]=true;report["candidate_statuses"]=statuses
    if argv.count==4{report["status"]="bindings_verified_no_gpu";report["finished_unix"]=Date().timeIntervalSince1970;try save();print("Bindings verified; no GPU execution");exit(0)}
    guard let device=MTLCreateSystemDefaultDevice(),let queue=device.makeCommandQueue() else{throw failure("Metal unavailable")}
    report["device"]=device.name;report["registry_id"]=String(device.registryID);report["os"]=ProcessInfo.processInfo.operatingSystemVersionString;report["gpu_executed"]=true
    let guardWord:UInt32=0x7fc01234,guardCount=16
    func makeBuffer(_ words:[UInt32])throws->MTLBuffer{guard let b=words.withUnsafeBytes({device.makeBuffer(bytes:$0.baseAddress!,length:$0.count,options:.storageModeShared)})else{throw failure("Allocation failed")};return b}
    var results=[[String:Any]]();var referencePassed=true;var allCandidatesPassed=true
    for (pi,p) in probes.enumerated() {
        let operation=p["operation"] as! String, function=p["function"] as! String,cases=p["cases"] as! [[String:Any]]
        var pipelines=[String:MTLComputePipelineState]()
        for label in ["reference","candidate"] {if let data=captured[pi][label] {let library=try device.makeLibrary(data:data.withUnsafeBytes{DispatchData(bytes:$0)});guard let f=library.makeFunction(name:function)else{throw failure("Function missing")};let pipeline=try device.makeComputePipelineState(function:f);guard pipeline.threadExecutionWidth==32 else{throw failure("Expected 32-lane SIMD")};pipelines[label]=pipeline}}
        if pipelines["candidate"]==nil{allCandidatesPassed=false}
        for (ci,c) in cases.enumerated() {
            let n=operation=="sum" ? c["n"] as! Int : (c["vectors_per_row"] as! Int)*(c["rows"] as! Int)*4
            let x=(0..<max(1,n)).map{Float(($0*13)%37-18)/16}
            let nb=operation=="sum" ? 1 : c["vectors_per_row"] as! Int
            let y=(0..<nb*4).map{Float(($0*7)%23-11)/16}
            var expected=[Float]()
            if operation=="sum" {if n>0{expected=[Float(x.prefix(n).reduce(0.0){$0+Double($1)})]}}
            else {expected=(0..<n).map{x[$0]*y[$0%(nb*4)]}}
            let outputCount=operation=="sum" ? 1:n
            let xWords=[UInt32](repeating:guardWord,count:guardCount)+x.map{$0.bitPattern}+[UInt32](repeating:guardWord,count:guardCount)
            let yWords=[UInt32](repeating:guardWord,count:guardCount)+y.map{$0.bitPattern}+[UInt32](repeating:guardWord,count:guardCount)
            var referenceBits:[UInt32]?=nil
            for label in ["reference","candidate"] {
                guard let pipeline=pipelines[label] else{continue}
                let xb=try makeBuffer(xWords),yb=try makeBuffer(yWords),out=try makeBuffer([UInt32](repeating:guardWord,count:guardCount+outputCount+guardCount))
                guard let command=queue.makeCommandBuffer(),let encoder=command.makeComputeCommandEncoder()else{throw failure("Command allocation")}
                encoder.setComputePipelineState(pipeline)
                if operation=="sum" {
                    var count=UInt64(n);let threads=c["threads"] as! Int
                    guard threads<=pipeline.maxTotalThreadsPerThreadgroup,threads%32==0 else{throw failure("Unsupported group size")}
                    encoder.setBytes(&count,length:8,index:0);encoder.setBuffer(xb,offset:guardCount*4,index:1);encoder.setBuffer(out,offset:guardCount*4,index:2);encoder.setThreadgroupMemoryLength(max(16,(threads/32)*4),index:0)
                    encoder.dispatchThreadgroups(MTLSize(width:1,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:threads,height:1,depth:1))
                }else{
                    var count=Int64(nb);encoder.setBuffer(xb,offset:guardCount*4,index:0);encoder.setBuffer(yb,offset:guardCount*4,index:1);encoder.setBuffer(out,offset:guardCount*4,index:2);encoder.setBytes(&count,length:8,index:3)
                    encoder.dispatchThreads(MTLSize(width:n/4,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:32,height:1,depth:1))
                }
                encoder.endEncoding();command.commit();command.waitUntilCompleted();guard command.status == .completed,command.error==nil else{throw failure("GPU command failed")}
                let words=Array(UnsafeBufferPointer(start:out.contents().bindMemory(to:UInt32.self,capacity:guardCount+outputCount+guardCount),count:guardCount+outputCount+guardCount))
                var errors=0,nonfinite=0,canaryErrors=0
                for i in 0..<words.count {
                    if i>=guardCount && i<guardCount+expected.count{let f=Float(bitPattern:words[i]);if !f.isFinite{nonfinite+=1};if f.bitPattern != expected[i-guardCount].bitPattern{errors+=1}}
                    else if words[i] != guardWord{canaryErrors+=1}
                }
                let xData=Data(bytes:xb.contents(),count:xb.length),yData=Data(bytes:yb.contents(),count:yb.length)
                let inputChanged=xData != xWords.withUnsafeBytes{Data($0)} || yData != yWords.withUnsafeBytes{Data($0)}
                let passed=errors==0 && nonfinite==0 && canaryErrors==0 && !inputChanged
                if label=="reference"{referencePassed = referencePassed && passed;referenceBits=words}else{allCandidatesPassed = allCandidatesPassed && passed}
                results.append(["probe":p["id"]!,"case_index":ci,"case":c,"variant":label,"output_elements":expected.count,"oracle_bit_mismatches":errors,"nonfinite_values":nonfinite,"canary_mutations":canaryErrors,"input_mutation":inputChanged,"reference_buffer_equal":label=="reference" || referenceBits==words,"output_sha256":words.withUnsafeBytes{digest(Data($0))},"input_sha256":digest(xData),"broadcast_source_sha256":digest(yData),"passed":passed])
                report["cases"]=results;try save()
            }
        }
    }
    report["reference_passed"]=referencePassed;report["candidate_passed"]=allCandidatesPassed;report["passed"]=referencePassed && allCandidatesPassed;report["numerical"]="completed_fixed_cases";report["finished_unix"]=Date().timeIntervalSince1970;try save()
    guard referencePassed else{throw failure("Original source reference failed independent oracle")}
    print("Reference passed: \(referencePassed); candidate passed: \(allCandidatesPassed). No speed claim.")
    if !allCandidatesPassed{exit(1)}
}catch{report["passed"]=false;report["error"]=String(describing:error);report["finished_unix"]=Date().timeIntervalSince1970;try? save();fputs("\(error)\n",stderr);exit(1)}
