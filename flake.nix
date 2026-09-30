{
  description = "agent-rl — a plain-English Agent RL handbook, built with mdBook";

  inputs = {
    # nixos-unstable is required: mdBook >= 0.5 renamed the book.toml key
    # `curly-quotes` to `smart-punctuation`, and this book uses the 0.5 spelling.
    # nixpkgs 24.05/24.11 ship mdBook 0.4.x and will refuse to deserialize it.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs { inherit system; }));
    in
    {
      formatter = forAllSystems (pkgs: pkgs.nixfmt-rfc-style);

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            pkgs.mdbook # the only tool the book actually needs
            pkgs.git
            pkgs.ripgrep # search across src/
            pkgs.fd # find a chapter by slug
            pkgs.jq
          ];

          shellHook = ''
            echo "agent-rl dev shell — mdbook $(mdbook --version | sed 's/^mdbook v//') ($(uname -m) $(uname -s))"
            echo "  mdbook serve --open   # live preview at http://localhost:3000"
            echo "  mdbook build          # writes the site to book/"
            echo "  nix build             # reproducible build; result/ is the whole site"
          '';
        };
      });

      # Reproducible build of the whole site: `nix build` → result/index.html.
      # book/ is excluded from the source so a stale build never feeds the next one.
      packages = forAllSystems (
        pkgs:
        let
          src = pkgs.lib.cleanSourceWith {
            src = ./.;
            filter =
              path: type:
              let
                base = builtins.baseNameOf (toString path);
              in
              !(builtins.elem base [
                "book"
                "result"
                ".direnv"
                ".git"
                ".DS_Store"
              ]);
          };
        in
        {
          default = pkgs.stdenvNoCC.mkDerivation {
            pname = "agent-rl-book";
            version = "unstable";
            inherit src;
            nativeBuildInputs = [ pkgs.mdbook ];
            buildPhase = ''
              runHook preBuild
              mdbook build --dest-dir "$out"
              runHook postBuild
            '';
            dontInstall = true;
          };
        }
      );
    };
}
