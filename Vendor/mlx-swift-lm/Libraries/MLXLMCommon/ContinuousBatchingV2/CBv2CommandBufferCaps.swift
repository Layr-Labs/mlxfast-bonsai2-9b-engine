// CBv2CommandBufferCaps.swift
//
// Per-phase Metal command-buffer commit caps. The vendored MLX build loads
// with large caps (2000 ops / 640 MB per buffer), which suit long prefill
// forwards. A decode step encodes small, dependent rounds, which run best
// with the stock caps the Metal device picks for its GPU. The engine loop
// selects the stock caps for the duration of each decode step and restores
// the large ones when the step ends.
//
// Commit points only decide where one command buffer ends and the next one
// starts on the same queue. Kernels, their inputs and their order do not
// change, so outputs are bit-identical under either setting.

import Darwin

enum CBv2CommandBufferCaps {
    private typealias Setter = @convention(c) (Int32) -> Int32

    /// `mlxfast_command_buffer_caps` from the vendored MLX build; nil (no
    /// switching, large caps everywhere) if the symbol is absent.
    private static let setter: Setter? = {
        guard
            let symbol = dlsym(
                UnsafeMutableRawPointer(bitPattern: -2),  // RTLD_DEFAULT
                "mlxfast_command_buffer_caps")
        else { return nil }
        return unsafeBitCast(symbol, to: Setter.self)
    }()

    /// Stock caps when `decode`, large prefill caps otherwise.
    static func select(decode: Bool) {
        _ = setter?(decode ? 1 : 0)
    }
}
