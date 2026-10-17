{
  description = "Los Podadores automation, RLlib cluster, and ROS workspace";

  inputs = {
    nix-ros-overlay.url = "github:lopsided98/nix-ros-overlay/master";
    nixpkgs.follows = "nix-ros-overlay/nixpkgs"; # IMPORTANT: matches binary cache & overlay compatibility
    nixpkgs-unstable.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  nixConfig = {
    extra-substituters = [
      "https://ros.cachix.org"
      "https://attic.iid.ciirc.cvut.cz/ros"
    ];
    extra-trusted-public-keys = [
      "ros.cachix.org-1:dSyZxI8geDCJrwgvCOHDoAfOm5sV1wCPjBkKL+38Rvo="
      "ros:JR95vUYsShSqfA1VTYoFt1Nz6uXasm5QrcOsGry9f6Q="
    ];
  };

  outputs =
    {
      self,
      nixpkgs,
      nixpkgs-unstable,
      flake-utils,
      nix-ros-overlay,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [ nix-ros-overlay.overlays.default ];
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

        colconPkg = pkgs.colcon.withExtensions [ pkgs.python3Packages.colcon-alias ];

        rosCoreEnvJazzy = with pkgs.rosPackages.jazzy; buildEnv {
          underlay = true;
          paths = [
            ros-core
            geometry-msgs
            sensor-msgs
            std-msgs
            rclpy
            rclcpp
            ros2launch
            ros2cli-common-extensions
          ];
        };

        rosDesktopEnvJazzy = with pkgs.rosPackages.jazzy; buildEnv {
          underlay = true;
          paths = [
            desktop
          ];
        };

        rosCoreEnvHumble = with pkgs.rosPackages.humble; buildEnv {
          underlay = true;
          paths = [
            ros-core
            geometry-msgs
            sensor-msgs
            std-msgs
            rclpy
            rclcpp
            ros2launch
            ros2cli-common-extensions
          ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          name = "los-podadores-dev";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
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

              colconPkg
              rosCoreEnvJazzy
            ])
            (with unstable; [

            ])
          ];

          buildInputs = [ pkgs.bashInteractive ];

          shellHook = ''
            export PATH="${pkgs.cmake}:${pkgs.ninja}:$PATH"
            # Prevent nix ros-env python3.14 packages from contaminating Python 3.12 virtualenvs
            unset PYTHONPATH
          '';

          env = { };
        };

        devShells.ros = pkgs.mkShell {
          name = "ros2-jazzy-dev";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
          packages = [
            colconPkg
            rosCoreEnvJazzy
          ];
          buildInputs = [ pkgs.bashInteractive ];
        };

        devShells.ros-desktop = pkgs.mkShell {
          name = "ros2-jazzy-desktop";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
          packages = [
            colconPkg
            rosDesktopEnvJazzy
          ];
          buildInputs = [ pkgs.bashInteractive ];
        };

        devShells.ros-humble = pkgs.mkShell {
          name = "ros2-humble-dev";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
          packages = [
            colconPkg
            rosCoreEnvHumble
          ];
          buildInputs = [ pkgs.bashInteractive ];
        };
      }
    );
}
