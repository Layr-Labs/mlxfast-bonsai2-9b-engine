import Foundation
import MLXFastCore
import Testing

// Contract tests for `fixtures/bonsai2_9b_mlx_v1_track.json`.
//
// The target is the Ternary Bonsai 2 9B MLX pack, and the track is DISARMED
// until the ranked box is calibrated for it. So:
//
// - the target pins name `prism-ml/Ternary-Bonsai-2-9B-mlx-2bit` @ f4e28a59
//   (`prism_hadamard_qwen35`, affine 2-bit group 128, signed block-4096
//   Hadamard), and the pack's own config and tensor inventory are checked in;
// - the head pins the pack's own `mtp/` folder of the same revision: a
//   standalone 4-bit `qwen3_5_mtp` drafter, two files;
// - `official_scoring_enabled` is true. The golden pins name the pack: they
//   were recorded on the ranked box against the pack, under new R2 keys. This
//   suite checks their shape only;
// - the geometry is the Qwen 3.5 9B backbone's, read off its own published
//   `config.json` (`fixtures/qwen35_9b_base_config.json`), and the pack's own
//   config and the compiled constants agree with it.
//
// This suite asserts that state as a COHERENT WHOLE.
//
// This is a laptop-side JSON-shape test only. It does not exercise benchd's
// Rust contract parsers; those live in the bench repository.

@Suite("Bonsai 2 9B track contract fixture")
struct Bonsai2TrackFixtureTests {

    @Test("fixture parses as a JSON object")
    func fixtureParses() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["schema_version"] as? Int == 1)
    }

    @Test("track_id is pinned to the leaderboard / R2-prefix identity")
    func trackIdIsPinned() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["track_id"] as? String == "bonsai2-9b-mlx-v1")
    }

    @Test("benchmark_name matches the Yukon manifest name")
    func benchmarkNameMatchesManifest() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["benchmark_name"] as? String == "mlxfast-bonsai2-9b")
    }

    @Test("track_id is substring-clean against every retired MTP name")
    func trackIdIsCleanAgainstRetiredNames() throws {
        let object = try bonsai2TrackContractObject()
        let trackId = try #require(object["track_id"] as? String)
        let retired = [
            "MLXFAST_MTP_",
            "mtp-ranked",
            "measure-mtp-job",
            "mtp-weights",
            "laguna-xs-2.1-mtp",
            "gemma4-31b-it",
        ]
        for name in retired {
            #expect(
                !trackId.contains(name),
                "track_id must not substring-collide with retired name \(name)")
            #expect(
                !name.contains(trackId),
                "retired name \(name) must not substring-collide with track_id")
        }
    }

    /// SCORING IS ARMED, AND THE GOLDEN PINS NAME THE PACK. The ranked box
    /// recorded the three live goldens against the pack, under new R2 keys in
    /// their own folder. This test checks the SHAPE of the pins: the fixture
    /// pins them as the timed pool, the hidden oracle is the first live golden,
    /// and every per-depth key pins its own oracle, recorded at that depth
    /// through the free run the candidate leg replays (David 2026-10-09: "do
    /// per depth oracles"). Scoring was switched on in its own change, after
    /// the box was calibrated for the pack.
    @Test("scoring is armed and the golden pins keep their shape")
    func scoringIsArmedAndTheGoldenPinsKeepTheirShape() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["official_scoring_enabled"] as? Bool == true)
        let live = try #require(object["live_goldens"] as? [String])
        #expect(live == ["pool-9", "pool-10", "pool-11"])
        let pool = try #require(object["timed_prompt_pool"] as? [[String: Any]])
        #expect(pool.count == live.count)
        // The pack's goldens live in their own folder of the append-only
        // bucket, named for the pack's revision; the stand-in's keys stay.
        let folder = "correctness_prompts/bonsai2-9b-mlx-v1/b94a52f3/"
        var byName: [String: [String: Any]] = [:]
        for entry in pool {
            let path = try #require(entry["r2_path"] as? String)
            #expect(path.hasPrefix(folder))
            #expect(path.hasSuffix(".golden.json"))
            let name = String(path.dropFirst(folder.count).dropLast(".golden.json".count))
            let sha = try #require(entry["sha256"] as? String)
            #expect(sha.count == 64 && sha.allSatisfy { "0123456789abcdef".contains($0) })
            #expect((entry["bytes"] as? Int ?? 0) > 0)
            byName[name] = entry
        }
        #expect(Set(byName.keys) == Set(live))
        #expect(Set(pool.compactMap { $0["sha256"] as? String }).count == pool.count)
        #expect(object["live_golden"] == nil)
        // ROOT level -- benchd's `hidden_correctness_golden_pin_from_contract`
        // reads this key directly off the contract root.
        let golden = try #require(object["hidden_correctness_golden"] as? [String: Any])
        let first = try #require(byName[live[0]])
        #expect(golden["sha256"] as? String == first["sha256"] as? String)
        #expect(golden["bytes"] as? Int == first["bytes"] as? Int)
        let oracles = try #require(
            object["live_golden_speculative"] as? [String: [String: [String: Any]]])
        let head = try #require(object["mtp_head"] as? [String: Any])
        let depths = try #require(head["permitted_draft_depths"] as? [Int])
        let drafter = try #require(object["dflash_drafter"] as? [String: Any])
        let dflashDepths = try #require(drafter["permitted_draft_depths"] as? [Int])
        #expect(
            Set(oracles.keys)
                == Set(depths.map { "mtp\($0)" } + dflashDepths.map { "dflash\($0)" }))
        // Each depth pins its OWN oracle per live golden, recorded at that depth
        // (tools/record-goldens.sh), under its own R2 key.
        for (key, perLive) in oracles {
            #expect(Set(perLive.keys) == Set(live), Comment(rawValue: key))
            for name in live {
                let pin = try #require(perLive[name])
                #expect(
                    pin["r2_path"] as? String
                        == "\(folder)\(name).\(key).golden.json",
                    Comment(rawValue: "\(key).\(name)"))
                let sha = try #require(pin["sha256"] as? String)
                #expect(sha.count == 64, Comment(rawValue: "\(key).\(name)"))
                #expect((pin["bytes"] as? Int ?? 0) > 0, Comment(rawValue: "\(key).\(name)"))
            }
        }
        #expect(object["hidden_material"] == nil)
    }

    /// MODE FENCE. This track has TWO speculative arms, the MTP head and the
    /// DFlash 2 drafter. `serial` must be present because the control leg is
    /// pinned serial. `dspark` must stay absent.
    @Test("allowed_modes declares serial, mtp and dflash")
    func allowedModesDeclaresSerialMTPAndDFlash() throws {
        let object = try bonsai2TrackContractObject()
        let modes = try #require(object["allowed_modes"] as? [String])
        #expect(modes == ["serial", "mtp", "dflash"])
        let drafter = try #require(object["dflash_drafter"] as? [String: Any])
        #expect(drafter["permitted_draft_depths"] as? [Int] == Array(1 ... 15))
        #expect(drafter["block_size"] as? Int == 16)
        let protocolBlock = try #require(object["protocol"] as? [String: Any])
        let envelope = try #require(
            protocolBlock["dflash_envelope_constants_from_darkbloom"] as? [String: Any])
        #expect(envelope["max_draft_tokens"] as? Int == 15)
    }

    @Test("the MTP head declares draft depths 1 to 6")
    func theHeadDeclaresItsPermittedDraftDepths() throws {
        let object = try bonsai2TrackContractObject()
        let head = try #require(object["mtp_head"] as? [String: Any])
        #expect(head["permitted_draft_depths"] as? [Int] == Array(1 ... 6))
    }

    /// THE COMBINE. Three pairs, and the score is the MEAN of the per-pair
    /// composites (David 2026-10-06, the same rule as the Nemotron track).
    @Test("official scoring is three pairs combined by the mean")
    func officialPairsCombineByTheMean() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["official_pairs"] as? Int == 3)
        #expect(object["official_pair_combine"] as? String == "mean")
    }

    @Test("kv_backend is pinned explicitly to contiguous")
    func kvBackendIsContiguous() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["kv_backend"] as? String == "contiguous")
    }

    @Test("scored_batch_size is pinned to the ruled single-stream width")
    func scoredBatchSizeIsOne() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["scored_batch_size"] as? Int == 1)
    }

    @Test("scored_exponents equals the ruled certify pair, exact field names")
    func scoredExponentsMatchesRuledPair() throws {
        let object = try bonsai2TrackContractObject()
        let exponents = try #require(object["scored_exponents"] as? [String: Any])
        // Field names must match benchd's `DeclaredScoredExponents` struct
        // exactly -- NOT the shorthand "prefill"/"decode" spelling, which
        // `ScoredExponents::certify` would treat as absent.
        let prefill = try #require(exponents["prefill_gain_exponent"] as? Double)
        let decode = try #require(exponents["decode_gain_exponent"] as? Double)
        #expect(prefill == 0.25)
        #expect(decode == 0.75)
        #expect(
            exponents.count == 2,
            "scored_exponents must carry exactly the certify pair, no extra keys")
    }

    @Test("the contract and the compiled constants agree on the scored window")
    func contractAgreesWithCompiledConstants() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["vocab_size"] as? Int == MLXFastConstants.vocabSize)
        #expect(object["num_hidden_layers"] as? Int == MLXFastConstants.numHiddenLayers)
        #expect(
            object["golden_model_type"] as? String
                == MLXFastConstants.requiredGoldenModelType)
        #expect(object["seed_tokens"] as? Int == MLXFastConstants.benchmarkDecodeSeedTokens)
        #expect(object["correctness_steps"] as? Int == MLXFastConstants.correctnessSteps)
        #expect(
            object["benchmark_decode_steps"] as? Int == MLXFastConstants.benchmarkDecodeSteps)
        #expect(
            object["local_submit_benchmark_decode_steps"] as? Int
                == MLXFastConstants.localSubmitBenchmarkDecodeSteps)
        #expect(object["score_decode_weight"] as? Double == MLXFastConstants.scoreDecodeWeight)
        #expect(object["score_prefill_weight"] as? Double == MLXFastConstants.scorePrefillWeight)
        #expect(
            object["decode_speedup_floor"] as? Double
                == MLXFastConstants.scoreDecodeSpeedupFloor)
        #expect(
            object["prefill_speedup_floor"] as? Double
                == MLXFastConstants.scorePrefillSpeedupFloor)
    }

    /// THE TARGET IS THE PACK. The repository and the revision name the
    /// Ternary Bonsai 2 9B MLX pack, and the compiled constants carry the same
    /// pin, so a golden recorded against any other checkpoint is refused. The
    /// checkpoint facts are the pack's own config and tensor inventory.
    @Test("the target pins name the pack, matching the compiled constants")
    func targetPinsAreThePack() throws {
        let object = try bonsai2TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        #expect(target["stand_in"] == nil)
        #expect(target["pending_pack"] == nil)
        #expect(target["upstream_model_id"] as? String == bonsai2PackRepository)
        #expect(target["upstream_revision"] as? String == bonsai2PackRevision)
        #expect(MLXFastConstants.pendingOrganizerSentinel == bonsai2PendingSentinel)
        #expect(MLXFastConstants.referenceModelRepository == bonsai2PackRepository)
        #expect(MLXFastConstants.referenceModelRevision == bonsai2PackRevision)
        #expect(target["model_type"] as? String == "prism_hadamard_qwen35")
        #expect(object["golden_model_type"] as? String == "prism_hadamard_qwen35")

        let configData = try Data(contentsOf: bonsai2PackConfigFixtureURL)
        #expect(sha256Hex(configData) == bonsai2PackConfigSHA256)
        #expect(configData.count == bonsai2PackConfigByteCount)
        let config = try bonsai2PackConfigObject()
        #expect(target["config_fixture_path"] as? String == "fixtures/bonsai2_9b_config.json")
        #expect(config["model_type"] as? String == target["model_type"] as? String)
        #expect(config["base_model_type"] as? String == target["base_model_type"] as? String)
        #expect(config["schema_version"] as? Int == target["schema_version"] as? Int)
        #expect(
            config["gdn_activation_layout"] as? String
                == target["gdn_activation_layout"] as? String)
        #expect(config["tensor_namespace"] as? String == target["tensor_namespace"] as? String)
        #expect(config["tie_word_embeddings"] as? Bool == false)

        let inventory = try bonsai2PackInventoryObject()
        #expect(
            target["tensor_inventory_path"] as? String
                == "fixtures/bonsai2_9b_tensor_inventory.json")
        let source = try #require(inventory["source"] as? [String: Any])
        #expect(source["upstream_model_id"] as? String == bonsai2PackRepository)
        #expect(source["upstream_revision"] as? String == bonsai2PackRevision)
        #expect(source["config_sha256"] as? String == sha256Hex(configData))
        #expect(source["config_bytes"] as? Int == configData.count)
        let summary = try #require(inventory["summary"] as? [String: Any])
        #expect(target["raw_checkpoint_tensor_count"] as? Int == summary["tensor_count"] as? Int)
        #expect(target["raw_checkpoint_shard_count"] as? Int == summary["shard_count"] as? Int)
        #expect(target["raw_checkpoint_total_size_bytes"] as? Int == summary["total_size_bytes"] as? Int)
        #expect(target["language_model_tensor_count"] as? Int == summary["language_model_tensor_count"] as? Int)
        #expect(target["vision_tower_tensor_count"] as? Int == summary["vision_tower_tensor_count"] as? Int)
        #expect(target["lm_head_tensor_count"] as? Int == summary["lm_head_tensor_count"] as? Int)
        #expect(summary["mtp_tensor_count"] as? Int == 0)

        // The inventory itself: two namespaces, and no head tensor in either.
        let tensors = try #require(inventory["tensors"] as? [String: Any])
        #expect(tensors.count == summary["tensor_count"] as? Int)
        let languageModel = tensors.keys.filter { $0.hasPrefix("language_model.") }
        let visionTower = tensors.keys.filter { $0.hasPrefix("vision_tower.") }
        #expect(languageModel.count == target["language_model_tensor_count"] as? Int)
        #expect(visionTower.count == target["vision_tower_tensor_count"] as? Int)
        #expect(languageModel.count + visionTower.count == tensors.count)
        #expect(
            languageModel.filter { $0.hasPrefix("language_model.lm_head.") }.count
                == target["lm_head_tensor_count"] as? Int)
        #expect(!tensors.keys.contains { $0.contains(".mtp.") || $0.hasPrefix("mtp.") })

        // The pack's tensors carry no head: its head is the `mtp/` folder.
        let components = try #require(target["components"] as? [String: Any])
        let published = try #require(config["components"] as? [String: Any])
        for key in ["text", "vision", "mtp"] {
            #expect(
                components[key] as? Bool == published[key] as? Bool,
                Comment(rawValue: "components.\(key)"))
        }
        #expect(components["text"] as? Bool == true)
        #expect(components["mtp"] as? Bool == false)
        #expect(components["vision"] as? Bool == true)
        #expect(target["vision_tower_loaded"] as? Bool == false)
        #expect(target["upstream_base_model"] as? String == bonsai2BaseRepository)
    }

    /// The pack has the backbone's geometry: its own `text_config` is the
    /// Qwen 3.5 9B config's, key for key, and its `vision_config` is the
    /// backbone's whole. The one difference is the head depth. The backbone
    /// declares one MTP layer; the pack declares 0, because its tensors carry
    /// no head.
    @Test("the pack's text_config is the Qwen 3.5 9B text_config, except the head depth")
    func packGeometryIsTheBackbone() throws {
        let packConfig = try bonsai2PackConfigObject()
        let baseConfig = try bonsai2BaseConfigObject()
        let pack = try #require(packConfig["text_config"] as? [String: Any])
        let base = try #require(baseConfig["text_config"] as? [String: Any])
        for key in [
            "num_hidden_layers", "hidden_size", "intermediate_size", "vocab_size",
            "max_position_embeddings", "full_attention_interval", "num_attention_heads",
            "num_key_value_heads", "head_dim", "linear_num_key_heads",
            "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
            "linear_conv_kernel_dim", "eos_token_id",
        ] {
            #expect(
                pack[key] as? Int == base[key] as? Int,
                Comment(rawValue: "text_config.\(key)"))
        }
        #expect(pack["layer_types"] as? [String] == base["layer_types"] as? [String])
        #expect(pack["mamba_ssm_dtype"] as? String == base["mamba_ssm_dtype"] as? String)
        #expect(
            pack["rope_parameters"] as? NSDictionary == base["rope_parameters"] as? NSDictionary)
        #expect(
            packConfig["vision_config"] as? NSDictionary
                == baseConfig["vision_config"] as? NSDictionary)
        #expect(pack["mtp_num_hidden_layers"] as? Int == 0)
        #expect(base["mtp_num_hidden_layers"] as? Int == 1)
    }

    @Test("the geometry source is the base config fixture, byte for byte")
    func geometrySourceIsTheBaseConfig() throws {
        let data = try Data(contentsOf: bonsai2BaseConfigFixtureURL)
        #expect(sha256Hex(data) == bonsai2BaseConfigSHA256)
        #expect(data.count == bonsai2BaseConfigByteCount)

        let object = try bonsai2TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let source = try #require(target["geometry_source"] as? [String: Any])
        #expect(source["upstream_model_id"] as? String == bonsai2BaseRepository)
        #expect(source["upstream_revision"] as? String == bonsai2BaseRevision)
        #expect(source["config_path"] as? String == "fixtures/qwen35_9b_base_config.json")
        #expect(source["sha256"] as? String == bonsai2BaseConfigSHA256)
        #expect(source["bytes"] as? Int == bonsai2BaseConfigByteCount)
    }

    /// The `target` geometry is READ OFF the backbone's own config.json, so
    /// every geometry field is asserted against that file rather than against
    /// a second hand-written copy of the same numbers.
    @Test("target geometry matches the Qwen 3.5 9B config.json")
    func targetGeometryMatchesConfig() throws {
        let object = try bonsai2TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let config = try bonsai2BaseConfigObject()
        let text = try #require(config["text_config"] as? [String: Any])
        let rope = try #require(text["rope_parameters"] as? [String: Any])

        #expect(target["model_type"] as? String == "prism_hadamard_qwen35")
        #expect(target["base_model_type"] as? String == config["model_type"] as? String)
        #expect(target["text_model_type"] as? String == text["model_type"] as? String)

        for key in [
            "num_hidden_layers", "hidden_size", "intermediate_size", "vocab_size",
            "max_position_embeddings", "full_attention_interval",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "linear_num_key_heads", "linear_num_value_heads",
            "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim",
        ] {
            #expect(
                target[key] as? Int == text[key] as? Int,
                Comment(rawValue: "target.\(key) must be the config's own"))
        }
        #expect(target["attention_bias"] as? Bool == text["attention_bias"] as? Bool)
        #expect(target["attn_output_gate"] as? Bool == text["attn_output_gate"] as? Bool)
        #expect(target["tie_word_embeddings"] as? Bool == config["tie_word_embeddings"] as? Bool)
        #expect(target["rope_theta"] as? Int == rope["rope_theta"] as? Int)
        #expect(target["partial_rotary_factor"] as? Double == rope["partial_rotary_factor"] as? Double)
        #expect(target["eos_token_ids"] as? [Int] == [try #require(text["eos_token_id"] as? Int)])
        #expect(target["mamba_ssm_dtype"] as? String == text["mamba_ssm_dtype"] as? String)

        // The compiled constants and the backbone are one set.
        #expect(target["hidden_size"] as? Int == MLXFastConstants.hiddenSize)
        #expect(target["num_hidden_layers"] as? Int == MLXFastConstants.numHiddenLayers)
        #expect(target["vocab_size"] as? Int == MLXFastConstants.vocabSize)
        // THE HEAD IS UNTIED: `lm_head` is a real quantized module.
        #expect(target["tie_word_embeddings"] as? Bool == false)
    }

    @Test("the declared layer schedule is the config's own, and its summaries are derived")
    func layerScheduleSummariesAreDerived() throws {
        let object = try bonsai2TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let config = try bonsai2BaseConfigObject()
        let text = try #require(config["text_config"] as? [String: Any])
        let layerTypes = try #require(text["layer_types"] as? [String])
        let interval = try #require(text["full_attention_interval"] as? Int)

        #expect(layerTypes.count == target["num_hidden_layers"] as? Int)
        let counts = try #require(target["layer_type_counts"] as? [String: Int])
        for kind in ["full_attention", "linear_attention"] {
            #expect(
                counts[kind] == layerTypes.filter { $0 == kind }.count,
                Comment(rawValue: "layer_type_counts.\(kind)"))
        }
        #expect(counts.values.reduce(0, +) == layerTypes.count)

        let indices = try #require(target["full_attention_layer_indices"] as? [Int])
        #expect(indices == layerTypes.indices.filter { layerTypes[$0] == "full_attention" })
        // Every fourth layer, and nothing else.
        #expect(indices == (0 ..< layerTypes.count).filter { ($0 + 1) % interval == 0 })
    }

    /// The packing fields the contract declares are the pack config's own,
    /// and the config's `modules` list is the 9B schedule: one packed module
    /// for each projection of each layer, the head and the embedding, each at
    /// the contract's Hadamard block width.
    @Test("the contract's packing fields are the pack config's own")
    func packingFieldsAreThePackConfigs() throws {
        let object = try bonsai2TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let config = try bonsai2PackConfigObject()
        let quantization = try #require(target["quantization"] as? [String: Any])
        let published = try #require(config["quantization"] as? [String: Any])
        #expect(Set(published.keys) == ["mode", "bits", "group_size"])
        #expect(quantization["mode"] as? String == published["mode"] as? String)
        #expect(quantization["bits"] as? Int == published["bits"] as? Int)
        #expect(quantization["group_size"] as? Int == published["group_size"] as? Int)

        let hadamard = try #require(target["hadamard"] as? [String: Any])
        #expect(hadamard["manifest_path_in_checkpoint"] as? String == config["hadamard_config"] as? String)
        let modules = try #require(config["modules"] as? [[String: Any]])
        #expect(modules.count == hadamard["packed_module_count"] as? Int)
        #expect(modules.allSatisfy { $0["block"] as? Int == hadamard["block_size"] as? Int })
        let paths = modules.compactMap { $0["path"] as? String }
        #expect(paths.count == modules.count)
        #expect(Set(paths).count == paths.count)
        let embeddings = modules.filter { $0["embedding"] as? Bool == true }
        #expect(embeddings.map { $0["path"] as? String } == [hadamard["embedding_module"] as? String])

        let base = try bonsai2BaseConfigObject()
        let text = try #require(base["text_config"] as? [String: Any])
        let layerTypes = try #require(text["layer_types"] as? [String])
        var expected: Set<String> = ["lm_head", "model.embed_tokens"]
        for index in layerTypes.indices {
            let prefix = "model.layers.\(index)"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                expected.insert("\(prefix).mlp.\(projection)")
            }
            let mixer = layerTypes[index] == "full_attention"
                ? ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
                : ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"]
            for projection in mixer {
                expected.insert("\(prefix).\(projection)")
            }
        }
        #expect(Set(paths) == expected)
    }

    /// The packing is uniform affine 2-bit at group 128, with a signed
    /// block-4096 Hadamard transform. (The Ternary Bonsai 2 27B pack uses a
    /// 1,024-wide block; this pack does not.) The derived counts follow from
    /// the geometry: one packed module each for the embedding and the head,
    /// three for each MLP, three for each linear-attention mixer and four for
    /// each full-attention mixer.
    @Test("the packing is uniform 2-bit at block 4096 and its counts follow from the geometry")
    func packingFollowsFromTheGeometry() throws {
        let object = try bonsai2TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let quantization = try #require(target["quantization"] as? [String: Any])
        #expect(quantization["mode"] as? String == "affine")
        #expect(quantization["bits"] as? Int == 2)
        #expect(quantization["group_size"] as? Int == 128)
        #expect(quantization["scale_dtype"] as? String == "float16")
        #expect(quantization["bias_dtype"] as? String == "float16")
        #expect(quantization["mixed_precision"] as? Bool == false)

        let hadamard = try #require(target["hadamard"] as? [String: Any])
        #expect(hadamard["manifest_path_in_checkpoint"] as? String == "hadamard.json")
        #expect(hadamard["block_size"] as? Int == 4096)
        #expect(hadamard["sign_dtype"] as? String == "float32")
        #expect(hadamard["arithmetic_dtype"] as? String == "float32")
        #expect(hadamard["embedding_module"] as? String == "model.embed_tokens")

        let counts = try #require(target["layer_type_counts"] as? [String: Int])
        let linear = try #require(counts["linear_attention"])
        let full = try #require(counts["full_attention"])
        let packed = 2 + linear * (3 + 3) + full * (3 + 4)
        #expect(hadamard["packed_module_count"] as? Int == packed)
        #expect(hadamard["sign_vector_count"] as? Int == packed)
        // Nine top-level tensors and 32 per layer, of either kind.
        let layers = try #require(target["num_hidden_layers"] as? Int)
        #expect(target["language_model_tensor_count"] as? Int == 9 + 32 * layers)
        #expect(target["lm_head_tensor_count"] as? Int == 4)

        // The inventory agrees: one U32 code tensor and one sign vector per
        // packed module.
        let inventory = try bonsai2PackInventoryObject()
        let summary = try #require(inventory["summary"] as? [String: Any])
        #expect(summary["packed_module_count"] as? Int == packed)
        #expect(summary["sign_vector_count"] as? Int == packed)
        let dtypes = try #require(summary["dtype_counts"] as? [String: Int])
        #expect(dtypes["U32"] == packed)
    }

    /// THE HEAD IS THE PACK'S OWN DRAFTER: the `mtp/` folder of the target's
    /// repository and revision, a standalone `qwen3_5_mtp` export of 31
    /// tensors, re-quantized to 4-bit affine group-64, with no name prefix.
    /// What the geometry fixes is asserted too:
    /// one layer, full attention, sharing the target's embedding and output
    /// head. The pack's own `text_config` declares no head depth (0), so the
    /// depth is the backbone's one MTP layer, and the loader reads it off the
    /// head's tensors.
    @Test("mtp_head is the pack's own mtp/ drafter")
    func mtpHeadIsThePacksOwnDrafter() throws {
        let object = try bonsai2TrackContractObject()
        let head = try #require(object["mtp_head"] as? [String: Any])
        let target = try #require(object["target"] as? [String: Any])
        let config = try bonsai2BaseConfigObject()
        let text = try #require(config["text_config"] as? [String: Any])
        #expect(head["role"] as? String == "spec_decode_mtp_head")
        #expect(head["source"] as? String == "target_repository_folder")
        #expect(head["upstream_model_id"] as? String == bonsai2PackRepository)
        #expect(head["upstream_revision"] as? String == bonsai2PackRevision)
        #expect(head["upstream_model_id"] as? String == target["upstream_model_id"] as? String)
        #expect(head["upstream_revision"] as? String == target["upstream_revision"] as? String)
        #expect(head["upstream_folder"] as? String == "mtp")
        #expect(head["tensor_prefix"] as? String == "")
        #expect(head["tensor_count"] as? Int == 31)
        #expect(head["tensor_bytes"] as? Int == 136_881_152)
        #expect(head["block_size"] as? Int == 2)
        // The head is re-quantized to 4-bit affine group-64, so it declares a
        // quantization block and no single tensor_dtype.
        #expect(head["tensor_dtype"] == nil)
        let headQuant = try #require(head["quantization"] as? [String: Any])
        #expect(headQuant["mode"] as? String == "affine")
        #expect(headQuant["bits"] as? Int == 4)
        #expect(headQuant["group_size"] as? Int == 64)

        #expect(head["model_type"] as? String == "qwen3_5_mtp")
        #expect(head["num_hidden_layers"] as? Int == text["mtp_num_hidden_layers"] as? Int)
        #expect(head["num_hidden_layers"] as? Int == 1)
        #expect(target["mtp_num_hidden_layers"] as? Int == 0)
        #expect(head["layer_types"] as? [String] == ["full_attention"])
        #expect(head["use_dedicated_embeddings"] as? Bool
            == text["mtp_use_dedicated_embeddings"] as? Bool)
        #expect(head["shares_target_embed_tokens"] as? Bool == true)
        #expect(head["shares_target_lm_head"] as? Bool == true)
        #expect(head["manifest_path"] as? String == "fixtures/reference_bonsai2_9b_mtp_head.sha256")
        #expect(head["declaration_path"] as? String == "mtp-head.manifest.json")

        // The head's one shard holds the tensor bytes and its own header.
        let shard = try #require(
            try bonsai2HeadManifestRecords().first { $0.path == "model.safetensors" })
        let tensorBytes = try #require(head["tensor_bytes"] as? Int)
        #expect(shard.byteCount > tensorBytes)
    }

    @Test("permitted draft depths fit the declared MTP envelope")
    func permittedDraftDepthsFitTheEnvelope() throws {
        let object = try bonsai2TrackContractObject()
        let head = try #require(object["mtp_head"] as? [String: Any])
        let depths = try #require(head["permitted_draft_depths"] as? [Int])
        let protocolBlock = try #require(object["protocol"] as? [String: Any])
        let envelope = try #require(
            protocolBlock["mtp_envelope_constants_from_darkbloom"] as? [String: Any])
        let maxDraft = try #require(envelope["max_draft_tokens"] as? Int)

        #expect(!depths.isEmpty)
        #expect(depths == Array(1 ... maxDraft))
        #expect(envelope["max_speculative_batch"] as? Int == object["scored_batch_size"] as? Int)
    }

    /// Both manifests pin the pack's revision, and their header pins are
    /// summed from their records. The target manifest pins 22 files: one
    /// safetensors shard and no index, `hadamard.json`, and the pack's own
    /// tokenizer. The head manifest pins the 2 files of the `mtp/` folder,
    /// with paths relative to it, so no head record is a target record.
    @Test("both reference manifests pin the pack, and the head pins its mtp/ folder")
    func referenceManifestsPinThePack() throws {
        for url in [bonsai2ReferenceManifestURL, bonsai2HeadManifestURL] {
            let records = try bonsai2ManifestRecords(at: url)
            let pins = try bonsai2ManifestHeaderPins(at: url)
            #expect(!records.isEmpty, Comment(rawValue: url.lastPathComponent))
            #expect(pins.records == records.count, Comment(rawValue: url.lastPathComponent))
            #expect(
                pins.bytes == records.reduce(0) { $0 + $1.byteCount },
                Comment(rawValue: url.lastPathComponent))
            let text = try String(contentsOf: url, encoding: .utf8)
            #expect(
                text.contains("# Revision: \(bonsai2PackRevision)\n"),
                Comment(rawValue: url.lastPathComponent))
            #expect(text.contains(bonsai2PackRepository), Comment(rawValue: url.lastPathComponent))
        }
        let target = try bonsai2ReferenceManifestRecords()
        let head = try bonsai2HeadManifestRecords()
        #expect(target.count == 22)
        #expect(head.map(\.path) == ["config.json", "model.safetensors"])
        #expect(Set(target.map(\.sha256)).isDisjoint(with: head.map(\.sha256)))
        #expect(!target.contains { $0.path.hasPrefix("mtp/") })

        let paths = Set(target.map(\.path))
        #expect(target.filter { $0.path.hasSuffix(".safetensors") }.map(\.path) == ["model.safetensors"])
        #expect(!paths.contains("model.safetensors.index.json"))
        #expect(paths.contains("hadamard.json"))
        #expect(paths.contains("tokenizer.json"))
        #expect(paths.contains("chat_template.jinja"))

        let config = try #require(target.first { $0.path == "config.json" })
        #expect(config.sha256 == bonsai2PackConfigSHA256)
        #expect(config.byteCount == bonsai2PackConfigByteCount)
        let inventory = try bonsai2PackInventoryObject()
        let source = try #require(inventory["source"] as? [String: Any])
        let hadamard = try #require(target.first { $0.path == "hadamard.json" })
        #expect(source["hadamard_manifest_sha256"] as? String == hadamard.sha256)
        #expect(source["hadamard_manifest_bytes"] as? Int == hadamard.byteCount)

        let object = try bonsai2TrackContractObject()
        let contractTarget = try #require(object["target"] as? [String: Any])
        #expect(contractTarget["manifest_path"] as? String == "fixtures/reference_bonsai2_9b_2bit.sha256")
        let shard = try #require(target.first { $0.path == "model.safetensors" })
        #expect(shard.byteCount == contractTarget["raw_checkpoint_total_size_bytes"] as? Int)
    }

    /// The shipped declaration is pinned and names a permitted decoder. A
    /// Yukon promotion writes the winning arm to main, so the declaration may
    /// name either speculative arm the fixture permits -- the MTP head or the
    /// DFlash drafter -- at any depth in that arm's permitted set. A disabled
    /// arm is serial (depth 0). The test checks the shape, not one arm or one
    /// depth, and reads the permitted ranges from the fixture so it tracks
    /// `allowed_modes` instead of fencing out a legal DFlash declaration.
    @Test("the shipped head declaration is pinned and names a permitted decoder")
    func theShippedDeclarationIsPinnedMTP() throws {
        let declaration = try #require(
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: bonsai2HeadDeclarationURL)) as? [String: Any])
        #expect(declaration["source"] as? String == "pinned")
        let spec = try #require(declaration["spec"] as? [String: Any])
        let enabled = try #require(spec["enabled"] as? Bool)
        let depth = try #require(spec["num_speculative_tokens"] as? Int)
        #expect(declaration["max_bytes"] as? Int == 2_147_483_648)

        let contract = try bonsai2TrackContractObject()
        let mtpDepths = try #require(
            (contract["mtp_head"] as? [String: Any])?["permitted_draft_depths"] as? [Int])
        let dflashDepths = try #require(
            (contract["dflash_drafter"] as? [String: Any])?["permitted_draft_depths"] as? [Int])
        if enabled {
            let decoder = try #require(spec["decoder"] as? String)
            #expect(["mtp", "dflash"].contains(decoder))
            let permitted = decoder == "dflash" ? dflashDepths : mtpDepths
            #expect(
                permitted.contains(depth),
                Comment(rawValue: "depth \(depth) is not permitted for decoder \(decoder)"))
        } else {
            #expect(depth == 0)
        }
    }

    @Test("scoring_semantics records the ruled composite formula")
    func scoringSemanticsRecordsRuledFormula() throws {
        let object = try bonsai2TrackContractObject()
        let semantics = try #require(object["scoring_semantics"] as? [String: Any])
        let formula = try #require(semantics["formula"] as? String)
        #expect(formula.contains("0.25"))
        #expect(formula.contains("0.75"))
        #expect(formula.contains("prefill_gain"))
        #expect(formula.contains("decode_gain"))
    }

    /// The license is the pack's own, read off its published LICENSE file,
    /// and the one verified repository is the pack's.
    @Test("the license block is the pack's and verifies its repository")
    func licenseIsThePacks() throws {
        let object = try bonsai2TrackContractObject()
        let license = try #require(object["license"] as? [String: Any])
        #expect(license["spdx"] as? String == "Apache-2.0")
        let url = try #require(license["url"] as? String)
        #expect(url.contains(bonsai2PackRepository))
        #expect(url.contains(bonsai2PackRevision))
        #expect(license["verified_repositories"] as? [String] == [bonsai2PackRepository])
    }
}
