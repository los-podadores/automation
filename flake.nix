{
  description = "A flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-26.05";
    nixpkgs-unstable.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      nixpkgs,
      nixpkgs-unstable,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        unstable = import nixpkgs-unstable {
          inherit system;
          config.allowUnfree = true;
        };
        libs = with pkgs; [
          stdenv.cc.cc
          zlib
          glib
          libxcb
          libglvnd
          libGL
          SDL2
          SDL2_image
          SDL2_mixer
          SDL2_ttf
          alsa-lib
          libX11
          wayland
          libxkbcommon
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          LD_LIBRARY_PATH =
            with pkgs;
            lib.makeLibraryPath [
              stdenv.cc.cc
              zlib
              glib
              libxcb
              libglvnd
              libGL
              SDL2
              SDL2_image
              SDL2_mixer
              SDL2_ttf
              alsa-lib
              libX11
              wayland
              libxkbcommon
            ];
          packages = pkgs.lib.flatten [
            (with pkgs; [
              uv
              ruff
              bun

              pkg-config
              freetype
              sdl2-compat

              gcc
              cmake
              ninja
              bazel_8
            ])
            (with unstable; [

            ])
          ];

          buildInputs = [ pkgs.bashInteractive ];

          shellHook = ''
            export PATH="${pkgs.cmake}:${pkgs.ninja}:$PATH"
          '';

          env = {
          };
        };
      }
    );
}
