{pkgs ? import <nixpkgs> {}}: let
  my-python-packages = ps:
    with ps; [
      pandas
      requests
      torch
      numpy
      scikit-learn
      virtualenv
    ];
  my-python = pkgs.python3.withPackages my-python-packages;
in
  my-python.env
