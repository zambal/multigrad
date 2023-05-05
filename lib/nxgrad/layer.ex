defmodule Nxgrad.Layer do
  import Nx.Defn, only: [defn: 2]
  alias Nxgrad.{Activation, Layer}

  @derive {Nx.Container, containers: [:b, :ws], keep: [:afun, :activation]}
  defstruct [:b, :ws, :afun, :activation]

  @type t :: %Layer{b: Nx.tensor(), ws: Nx.tensor(), afun: Activation.afun(), activation: Activation.type}

  def from_config(nin, nout, opts \\ []) do
    type = Keyword.get(opts, :param_type, :f64)
    b = for _n <- 1..nout, do: 0.0
    b = Nx.tensor(b, type: type)
    ws = for _n <- 1..nout, do: for(_n <- 1..nin, do: Multigrad.rand_number())
    ws = Nx.tensor(ws, type: type)
    activation = Keyword.get(opts, :activation, :linear)
    afun = Activation.afun(activation)

    %Layer{b: b, ws: ws, afun: afun, activation: activation}
  end

  def from_parameters({b, ws, activation}) do
    type = :f64
    b = Nx.tensor(b, type: type)
    ws = Nx.tensor(ws, type: type)
    afun = Activation.afun(activation)

    %Layer{b: b, ws: ws, afun: afun, activation: activation}
  end

  def to_parameters(%Layer{b: b, ws: ws, activation: activation}) do
    b = Nx.to_list(b)
    ws = Nx.to_list(ws)
    {b, ws, activation}
  end

  defn run(xs, %Layer{b: b, ws: ws, afun: afun}) do
    xs
    |> Nx.dot([Nx.rank(xs) - 1], ws, [1])
    |> Nx.add(b)
    |> afun.()
  end
end
