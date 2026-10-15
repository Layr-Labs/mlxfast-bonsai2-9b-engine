import CryptoKit
import Foundation
import MLXFastCore
import Testing

/// Fixture locations and pins for the track target,
/// `bonsai2-9b-mlx-v1`.
///
/// The organizer sentinel a pin carries while nothing is published.
/// Exact-match only, never a prefix check.
let bonsai2PendingSentinel = "BONSAI-MLX-PENDING-ORGANIZER"

/// THE TARGET. `prism-ml/Ternary-Bonsai-2-9B-mlx-2bit` @ b94a52f3 is the
/// Ternary Bonsai 2 9B MLX pack. Its `config.json` is checked in VERBATIM as
/// `fixtures/bonsai2_9b_config.json`, and its tensor inventory (read off the
/// published shard header) as `fixtures/bonsai2_9b_tensor_inventory.json`. The
/// digest below is the config fixture's own digest and the target manifest's
/// `config.json` record; a test holds them together.
let bonsai2PackRepository = "prism-ml/Ternary-Bonsai-2-9B-mlx-2bit"
let bonsai2PackRevision = "b94a52f3ce9047919dce00392a0fc6b8a7968abe"
let bonsai2PackConfigSHA256 =
    "b028784e087316b7ce0fb7ae3d2d6eaeb0b15144dd3f234da4c5d57ef1ffbacb"
let bonsai2PackConfigByteCount = 30_202

/// The geometry source: the Qwen 3.5 9B backbone's own `config.json`, checked
/// in VERBATIM as `fixtures/qwen35_9b_base_config.json`. The digest below is
/// the fixture's own digest and the contract's `target.geometry_source` pin; a
/// test holds them together.
let bonsai2BaseRepository = "Qwen/Qwen3.5-9B"
let bonsai2BaseRevision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
let bonsai2BaseConfigSHA256 =
    "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05"
let bonsai2BaseConfigByteCount = 3_126

// Tests/MLXFastTests/Model/<this file> -> repository root is four levels up.
private let bonsai2RepositoryRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

let bonsai2BaseConfigFixtureURL = bonsai2RepositoryRoot
    .appendingPathComponent("fixtures/qwen35_9b_base_config.json")

let bonsai2PackConfigFixtureURL = bonsai2RepositoryRoot
    .appendingPathComponent("fixtures/bonsai2_9b_config.json")

let bonsai2PackInventoryFixtureURL = bonsai2RepositoryRoot
    .appendingPathComponent("fixtures/bonsai2_9b_tensor_inventory.json")

let bonsai2TrackContractURL = bonsai2RepositoryRoot
    .appendingPathComponent("fixtures/bonsai2_9b_mlx_v1_track.json")

let bonsai2ReferenceManifestURL = bonsai2RepositoryRoot
    .appendingPathComponent("fixtures/reference_bonsai2_9b_2bit.sha256")

let bonsai2HeadManifestURL = bonsai2RepositoryRoot
    .appendingPathComponent("fixtures/reference_bonsai2_9b_mtp_head.sha256")

let bonsai2HeadDeclarationURL = bonsai2RepositoryRoot
    .appendingPathComponent("mtp-head.manifest.json")

func bonsai2TrackContractObject() throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(
        with: try Data(contentsOf: bonsai2TrackContractURL)
    ) as? [String: Any] else {
        throw MLXFastError.invalidInput(
            "Bonsai 2 track contract fixture must be a JSON object"
        )
    }
    return object
}

func bonsai2BaseConfigObject() throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(
        with: try Data(contentsOf: bonsai2BaseConfigFixtureURL)
    ) as? [String: Any] else {
        throw MLXFastError.invalidInput(
            "Qwen 3.5 9B base config fixture must be a JSON object"
        )
    }
    return object
}

func bonsai2PackConfigObject() throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(
        with: try Data(contentsOf: bonsai2PackConfigFixtureURL)
    ) as? [String: Any] else {
        throw MLXFastError.invalidInput("Bonsai 2 9B pack config fixture must be a JSON object")
    }
    return object
}

func bonsai2PackInventoryObject() throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(
        with: try Data(contentsOf: bonsai2PackInventoryFixtureURL)
    ) as? [String: Any] else {
        throw MLXFastError.invalidInput("Bonsai 2 9B tensor inventory fixture must be a JSON object")
    }
    return object
}

/// One `<sha256> <byte_count> <relative_path>` record of a pinned manifest.
/// Comment and blank lines are skipped.
struct Bonsai2ManifestRecord: Hashable {
    let sha256: String
    let byteCount: Int
    let path: String
}

func bonsai2ManifestRecords(at url: URL) throws -> [Bonsai2ManifestRecord] {
    let text = try String(contentsOf: url, encoding: .utf8)
    return try text.split(separator: "\n").compactMap { rawLine -> Bonsai2ManifestRecord? in
        let line = rawLine.trimmingCharacters(in: .whitespaces)
        guard !line.isEmpty, !line.hasPrefix("#") else { return nil }
        let fields = line.split(separator: " ", omittingEmptySubsequences: true)
        guard fields.count == 3, let byteCount = Int(fields[1]) else {
            throw MLXFastError.invalidInput(
                "reference manifest record is not <sha256> <bytes> <path>: \(line)"
            )
        }
        return Bonsai2ManifestRecord(
            sha256: String(fields[0]),
            byteCount: byteCount,
            path: String(fields[2])
        )
    }
}

func bonsai2ReferenceManifestRecords() throws -> [Bonsai2ManifestRecord] {
    try bonsai2ManifestRecords(at: bonsai2ReferenceManifestURL)
}

func bonsai2HeadManifestRecords() throws -> [Bonsai2ManifestRecord] {
    try bonsai2ManifestRecords(at: bonsai2HeadManifestURL)
}

/// The two shape pins a manifest header declares, read back out of it. They
/// are asserted against the records they must be summed from: a count that is
/// not summed from the digests it must agree with is not a pin.
func bonsai2ManifestHeaderPins(at url: URL) throws -> (records: Int, bytes: Int) {
    let text = try String(contentsOf: url, encoding: .utf8)
    func pin(_ key: String) throws -> Int {
        guard let line = text.split(separator: "\n").first(where: { $0.contains(key) }),
              let value = line.split(separator: ":").last
                  .map({ $0.trimmingCharacters(in: .whitespaces) }),
              let number = Int(value)
        else {
            throw MLXFastError.invalidInput(
                "reference manifest header is missing \(key)"
            )
        }
        return number
    }
    return (
        try pin("MLXFAST_REFERENCE_MANIFEST_RECORDS"),
        try pin("MLXFAST_REFERENCE_MANIFEST_BYTES")
    )
}

/// Lowercase hex SHA256 of some bytes, the spelling every pin in this
/// repository is written in.
func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
