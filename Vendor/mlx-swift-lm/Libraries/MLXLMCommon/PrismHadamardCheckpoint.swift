// Copyright © 2026 Eigen Labs.
// Prism artifact contract: HF prism-ml/Ternary-Bonsai-2-27B-mlx-2bit@3f926b415992eaa2ae9dd7b573706494d6bbf787.
import Cmlx
import Foundation
import MLX
import MLXNN

// Fixed launch values for the qualified 16-row residual norm.
private enum PrismResidualNorm16 {
    static let shapes = [[16, 4096], [512], [512], [1, 16, 4096]]
    static let dtypes: [DType] = [.int8, .float32, .float32, .float32]

    private final class Launch {
        let original: MLXFast.MLXFastKernel
        let kernel: mlx_fast_metal_kernel
        let config: mlx_fast_metal_kernel_config

        init(_ original: MLXFast.MLXFastKernel, _ metadata: (String, String, String, Int)) {
            self.original = original
            let (name, source, header, threads) = metadata
            let inputNames = mlx_vector_string_new()
            let outputNames = mlx_vector_string_new()
            defer {
                mlx_vector_string_free(inputNames)
                mlx_vector_string_free(outputNames)
            }
            for name in ["x", "residual", "norm_weight", "epsilon", "signs", "verify_rows"] {
                mlx_vector_string_append_value(inputNames, name)
            }
            for name in ["codes", "ascale", "rsb", "residual_out"] {
                mlx_vector_string_append_value(outputNames, name)
            }
            kernel = name.withCString { name in
                source.withCString { source in
                    header.withCString { header in
                        mlx_fast_metal_kernel_new(
                            name, inputNames, outputNames, source, header, true, false)
                    }
                }
            }
            config = mlx_fast_metal_kernel_config_new()
            mlx_fast_metal_kernel_config_add_template_arg_int(config, "BLOCKS_PER_ROW", 1)
            mlx_fast_metal_kernel_config_set_grid(config, Int32(threads), 16, 1)
            mlx_fast_metal_kernel_config_set_thread_group(config, Int32(threads), 1, 1)
            for (shape, dtype) in zip(PrismResidualNorm16.shapes, PrismResidualNorm16.dtypes) {
                mlx_fast_metal_kernel_config_add_output_arg(
                    config, shape.map(Int32.init), shape.count, dtype.cmlxDtype)
            }
            mlx_fast_metal_kernel_config_set_verbose(config, false)
        }

        deinit {
            mlx_fast_metal_kernel_config_free(config)
            mlx_fast_metal_kernel_free(kernel)
        }

        func apply(_ inputs: [MLXArray]) -> [MLXArray] {
            let arrays = inputs + [HostConstants.int32([16])]
            let input = withExtendedLifetime(arrays) {
                mlx_vector_array_new_data(arrays.map { $0.ctx }, arrays.count)
            }
            defer { mlx_vector_array_free(input) }
            var output = mlx_vector_array_new()
            defer { mlx_vector_array_free(output) }
            mlx_fast_metal_kernel_apply(&output, kernel, input, config, StreamOrDevice.default.ctx)
            return (0 ..< mlx_vector_array_size(output)).map { index in
                var ctx = mlx_array_new()
                mlx_vector_array_get(&ctx, output, index)
                return MLXArray(ctx)
            }
        }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var definitions:
        [ObjectIdentifier: (MLXFast.MLXFastKernel, (String, String, String, Int))] = [:]
    nonisolated(unsafe) private static var launch: Launch?
    private static let installed: Void = {
        SignedBlockHadamard.rn16 = { original, metadata, inputs, outputShapes in
            let key = ObjectIdentifier(original)
            if let metadata {
                lock.lock()
                defer { lock.unlock() }
                definitions[key] = (original, metadata)
                return nil
            }
            guard outputShapes == shapes else { return nil }
            lock.lock()
            if launch == nil, let definition = definitions[key] {
                launch = Launch(definition.0, definition.1)
            }
            let current = launch?.original === original ? launch : nil
            lock.unlock()
            return current?.apply(inputs)
        }
    }()

    static func install() { _ = installed }
}

public enum PrismCheckpointError: Error, LocalizedError {
    case invalid(String)
    public var errorDescription: String? {
        if case .invalid(let detail) = self { return "Invalid Prism Hadamard checkpoint: \(detail)" }
        return nil
    }
}

/// Packed artifact declaration, separate from the underlying Qwen configuration.
public struct PrismHadamardCheckpointConfiguration: Decodable, Sendable {
    public struct Record: Decodable, Sendable {
        public let path: String
        public let block: Int
        public let embedding: Bool
        public let dtype: String
    }
    private struct Components: Decodable { let text: Bool; let vision: Bool; let mtp: Bool }
    private struct Quantization: Decodable { let bits: Int; let group_size: Int; let mode: String }
    private struct Text: Decodable { let mtp_num_hidden_layers: Int?; let num_experts: Int? }
    public let modules: [Record]
    public let hasVision: Bool
    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version", modelType = "model_type", baseType = "base_model_type"
        case components, modules, quantization, text = "text_config"
        case layout = "gdn_activation_layout", namespace = "tensor_namespace", metadata = "hadamard_config"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let components = try c.decode(Components.self, forKey: .components)
        let quantization = try c.decode(Quantization.self, forKey: .quantization)
        let text = try c.decode(Text.self, forKey: .text)
        guard try c.decode(Int.self, forKey: .schemaVersion) == 2,
            try c.decode(String.self, forKey: .modelType) == "prism_hadamard_qwen35",
            try c.decode(String.self, forKey: .baseType) == "qwen3_5",
            try c.decode(String.self, forKey: .layout) == "grouped",
            try c.decode(String.self, forKey: .namespace) == "mlx-vlm-qwen3_5",
            try c.decode(String.self, forKey: .metadata) == "hadamard.json",
            components.text, !components.mtp, (text.mtp_num_hidden_layers ?? 0) == 0,
            (text.num_experts ?? 0) == 0,
            quantization.bits == 2, quantization.group_size == 128, quantization.mode == "affine"
        else { throw PrismCheckpointError.invalid("unsupported architecture/packing contract") }
        modules = try c.decode([Record].self, forKey: .modules)
        hasVision = components.vision
        guard !modules.isEmpty, modules.count <= 10_000,
            Set(modules.map(\.path)).count == modules.count,
            modules.allSatisfy({ record in
                let parts = record.path.split(separator: ".", omittingEmptySubsequences: false)
                return !parts.isEmpty && parts.allSatisfy { !$0.isEmpty && $0.allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" } }
                    && [512, 1024, 2048, 4096].contains(record.block) && record.dtype == "float16"
                    && record.embedding == (record.path == "model.embed_tokens")
            }), modules.filter(\.embedding).count == 1
        else { throw PrismCheckpointError.invalid("invalid packed module declarations") }
    }
}

public protocol PrismHadamardLoading {
    var prismCheckpoint: PrismHadamardCheckpointConfiguration { get }
}

struct PrismHadamardCheckpoint {
    let configuration: PrismHadamardCheckpointConfiguration
    let transforms: PrismHadamardConfiguration

    init(directory: URL, configuration: PrismHadamardCheckpointConfiguration,
         weights: [String: MLXArray]) throws {
        self.configuration = configuration
        // Hugging Face snapshots normally symlink immutable blobs. Resolve the
        // fixed metadata filename before the bounded regular-file check.
        let file = directory.appendingPathComponent("hadamard.json").resolvingSymlinksInPath()
        let values = try file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values.isRegularFile == true,
            let size = values.fileSize, size > 0, size <= 8 * 1024 * 1024
        else { throw PrismCheckpointError.invalid("unsafe transform metadata") }
        transforms = try JSONDecoder().decode(PrismHadamardConfiguration.self, from: Data(contentsOf: file))
        let normalizers = weights.filter {
            $0.key.hasPrefix("language_model.model.layers.")
                && ($0.key.hasSuffix(".input_layernorm.weight") || $0.key.hasSuffix(".post_attention_layernorm.weight"))
        }
        guard !normalizers.isEmpty, normalizers.values.allSatisfy({ $0.dtype == .float32 }) else {
            throw PrismCheckpointError.invalid("schema-2 native state requires the published FP32 normalizers")
        }
        let forward = Set(configuration.modules.filter { !$0.embedding }.map { "language_model.\($0.path).weight" })
        let inverse = Set(configuration.modules.filter(\.embedding).map { "language_model.\($0.path).weight" })
        guard transforms.gdnVGrouped, Set(transforms.weightNames) == forward,
            Set(transforms.inverseWeightNames) == inverse,
            Set(weights.keys.filter { $0.hasSuffix(".signs") }) == Set(configuration.modules.map { "language_model.\($0.path).signs" }),
            !weights.keys.contains(where: { $0.contains(".mtp.") || $0.hasPrefix("mtp.") })
        else { throw PrismCheckpointError.invalid("transform manifest disagrees with tensors") }
        for record in configuration.modules {
            let name = "language_model.\(record.path)"
            guard let weight = weights[name + ".weight"], let scales = weights[name + ".scales"],
                let biases = weights[name + ".biases"], let signs = weights[name + ".signs"],
                weight.ndim == 2, weight.dtype == .uint32, weight.dim(1) > 0,
                scales.ndim == 2, scales.dtype == .float16, biases.dtype == .float16,
                scales.shape == biases.shape, scales.dim(0) == weight.dim(0),
                scales.dim(1) * 8 == weight.dim(1), signs.shape == [weight.dim(1) * 16],
                [.float16, .float32, .bfloat16].contains(signs.dtype),
                record.block == transforms.blockSize
            else { throw PrismCheckpointError.invalid("packed tensor shape/dtype mismatch") }
            let transform = try transforms.transform(forWidth: weight.dim(1) * 16)
            guard transform.matches(signs: signs.asType(.float32).asArray(Float.self)) else {
                throw PrismCheckpointError.invalid("tensor signs disagree with metadata")
            }
        }
    }

    func install(model: Module, weights: [String: MLXArray]) throws {
        PrismResidualNorm16.install()
        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        var replacements: [(String, Module)] = []
        for record in configuration.modules {
            let name = "language_model.\(record.path)"
            guard let module = leaves[name], let weight = weights[name + ".weight"],
                let scales = weights[name + ".scales"], let biases = weights[name + ".biases"]
            else { throw PrismCheckpointError.invalid("declared module is missing") }
            let transform = try transforms.transform(forWidth: weight.dim(1) * 16)
            let replacement: Module
            if record.embedding {
                guard let embedding = module as? Embedding,
                    embedding.shape.0 == weight.dim(0), embedding.shape.1 == transform.width
                else { throw PrismCheckpointError.invalid("embedding dimensions differ from model") }
                replacement = try HadamardQuantizedEmbedding(weight: weight, scales: scales,
                    biases: biases, groupSize: 128, bits: 2, transform: transform)
            } else {
                guard let linear = module as? Linear,
                    linear.shape.0 == weight.dim(0), linear.shape.1 == transform.width
                else { throw PrismCheckpointError.invalid("linear dimensions differ from model") }
                replacement = try HadamardQuantizedLinear(weight: weight, scales: scales,
                    biases: biases, groupSize: 128, bits: 2, transform: transform)
            }
            replacements.append((name, replacement))
        }
        model.update(modules: ModuleChildren.unflattened(replacements))
    }
}
