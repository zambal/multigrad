defmodule Axongrad.Layer do
  alias Axongrad.Activation

  def from_config(nin) do
    Axon.input("xs", shape: {nil, nin})
  end

  def from_config({model, params}, nin, nout, opts) do
    type = :f64
    label = Keyword.fetch!(opts, :label)
    activation = Keyword.get(opts, :activation, :linear)
    ws = for _n <- 1..nout, do: for(_n <- 1..nin, do: Multigrad.rand_number())
    ws = Nx.tensor(ws, type: type)
    params = Map.put(params, label, %{"kernel" => ws})

    model =
      model
      |> Axon.dense(nout, name: label)
      |> Activation.afun(activation).()

    {model, params}
  end

  def from_parameters({model, params}, {b, ws, activation}, opts) do
    type = :f64
    label = Keyword.fetch!(opts, :label)
    nout = length(b)
    b = Nx.tensor(b, type: type)
    ws = Nx.tensor(ws, type: type)
    params = Map.put(params, label, %{"bias" => b, "kernel" => ws})

    model =
      model
      |> Axon.dense(nout, name: label)
      |> Activation.afun(activation).()

    {model, params}
  end
end
