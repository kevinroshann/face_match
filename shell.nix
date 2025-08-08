{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = [
    pkgs.nodejs
    (pkgs.python3.withPackages (pypkgs: [
      pypkgs.numpy
      pypkgs.deepface
    ]))
  ];
}
