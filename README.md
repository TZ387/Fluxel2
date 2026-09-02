# Fluxel2

An experiment in porting [Fluxel](https://github.com/TZ387/Fluxel) — a browser-based simulator for light transport in
biological tissue (diffusion approximation) — into a Tauri desktop app, targeting Linux and Windows.

Fluxel itself is a static HTML/CSS/vanilla-JS app with no build step; this project reworks it as a proper desktop
build (TypeScript frontend + Rust backend via Tauri) rather than something you just open in a browser.

Work in progress — currently just the Tauri scaffold, porting hasn't started yet.

## Development

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer) is the recommended setup.
