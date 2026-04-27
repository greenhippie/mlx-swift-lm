// Copyright © 2025 Apple Inc.
// Tests for https://github.com/ml-explore/mlx-swift-lm/issues/157
//
// Qwen35.LanguageModel holds precomputedPositionIds and ropeDeltas as
// instance state. Without resetting between inference rounds, stale
// position IDs from round N cause a broadcast_shapes crash in round N+1
// when the sequence length differs.
//
// The same pattern affects Qwen3VL and GlmOcr.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

// MARK: - Qwen35 stale position state regression test

final class Qwen35PositionStateTests: XCTestCase {

    /// Build a minimal Qwen35Configuration via JSON so the Codable
    /// init (which sets all the defaults) is exercised.
    private func makeMinimalQwen35Config() throws -> Qwen35Configuration {
        let json = """
        {
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "vocab_size": 256,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "max_position_embeddings": 1024,
                "tie_word_embeddings": true,
                "attention_bias": false,
                "full_attention_interval": 4,
                "rope_parameters": {
                    "rope_type": "default",
                    "mrope_section": [2, 2, 2],
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 0.25
                }
            },
            "vision_config": {
                "model_type": "qwen3_5",
                "depth": 2,
                "hidden_size": 64,
                "intermediate_size": 128,
                "out_hidden_size": 64,
                "num_heads": 4,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "num_position_embeddings": 256
            },
            "image_token_id": 200,
            "video_token_id": 201,
            "vision_start_token_id": 202,
            "vision_end_token_id": 203
        }
        """
        let data = json.data(using: .utf8)!
        return try JSONDecoder().decode(Qwen35Configuration.self, from: data)
    }

    /// Regression test: two sequential text-only inferences via prepare()
    /// with different sequence lengths must not crash.
    ///
    /// Before the fix, the second prepare() would crash with:
    ///   [broadcast_shapes] Shapes (N) and (M) cannot be broadcast
    /// because precomputedPositionIds from round 1 (length M) were
    /// reused for round 2 (length N).
    ///
    /// Note: callAsFunction() is the per-token continuation path and must
    /// NOT reset position state — only prepare() resets at the start of
    /// each new inference round.
    func testSequentialPrepareDoesNotCrash() throws {
        let config = try makeMinimalQwen35Config()
        let model = Qwen35(config)

        // Round 1
        let tokens1 = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]
        let mask1 = MLXArray.ones([1, 5]).asType(.int8)
        let lmInput1 = LMInput(text: .init(tokens: tokens1, mask: mask1))
        let cache1 = model.newCache(parameters: nil)
        let result1 = try model.prepare(lmInput1, cache: cache1, windowSize: nil)

        switch result1 {
        case .logits(let output):
            eval(output.logits)
            XCTAssertEqual(output.logits.shape[1], 5)
        case .tokens:
            XCTFail("Expected .logits, got .tokens")
        }

        // Round 2 — different sequence length, fresh cache
        let tokens2 = MLXArray([1, 2, 3, 4, 5, 6, 7, 8])[.newAxis, .ellipsis]
        let mask2 = MLXArray.ones([1, 8]).asType(.int8)
        let lmInput2 = LMInput(text: .init(tokens: tokens2, mask: mask2))
        let cache2 = model.newCache(parameters: nil)
        let result2 = try model.prepare(lmInput2, cache: cache2, windowSize: nil)

        switch result2 {
        case .logits(let output):
            eval(output.logits)
            XCTAssertEqual(output.logits.shape[1], 8)
        case .tokens:
            XCTFail("Expected .logits, got .tokens")
        }
    }
}

// MARK: - TokenRing 2D prompt regression test (issue #168)

final class TokenRingTests: XCTestCase {

    /// VLM prompts are 2D [1, N]. TokenRing.loadPrompt() must flatten
    /// before using dim(0), otherwise it sees 1 instead of N tokens.
    /// See: https://github.com/ml-explore/mlx-swift-lm/issues/168
    func testLoadPrompt2DArray() {
        var ring = TokenRing(capacity: 64)

        // Simulate a VLM prompt: 2D [1, 10]
        let prompt2D = MLXArray(Array(1...10))[.newAxis, .ellipsis]
        XCTAssertEqual(prompt2D.shape, [1, 10])

        ring.loadPrompt(prompt2D)

        // Must have loaded all 10 tokens, not just 1
        XCTAssertEqual(ring.count, 10)

        // Valid tokens should contain all 10
        let valid = ring.validTokens!
        XCTAssertEqual(valid.dim(0), 10)
    }

    /// 1D prompts (LLM text-only) must still work correctly.
    func testLoadPrompt1DArray() {
        var ring = TokenRing(capacity: 64)

        let prompt1D = MLXArray(Array(1...5))
        XCTAssertEqual(prompt1D.shape, [5])

        ring.loadPrompt(prompt1D)
        XCTAssertEqual(ring.count, 5)
    }
}
