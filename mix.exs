defmodule Multigrad.MixProject do
  use Mix.Project

  def project do
    [
      app: :multigrad,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:crypto, :eex, :logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.5"},
      {:torchx, "~> 0.5.3"},
      {:jason, "~> 1.4"},
      {:axon, "~> 0.5"}
    ]
  end
end
