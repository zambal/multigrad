defmodule Axongrad do
  @behaviour Multigrad

  alias Axongrad.MLP

  def to_training_data(samples, batch_size) do
    Nxgrad.to_training_data(samples, batch_size)
  end

  def from_config(definition, opts) when length(definition) in [4, 5] do
    {:ok, MLP.from_config(definition, opts)}
  end

  def from_parameters(parameters) do
    {:ok, MLP.from_parameters(parameters)}
  end

  def to_parameters(model, _opts) do
    MLP.to_parameters(model)
  end

  def train({model, _loss}, data, rate) do
    Enum.reduce(data, {model, 0}, fn {xs, ys}, {model, _loss} ->
      #xs = Nx.backend_copy(xs)
      #ys = Nx.backend_copy(ys)

      #run(model, xs, ys, rate)
    end)
  end

end
