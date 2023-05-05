defmodule Nxgrad do
  @behaviour Multigrad

  require Nx
  import Nx.Defn, only: [defn: 2, defnp: 2]
  alias Nxgrad.{Layer, MLP, Neuron}

  defmacro __using__(_) do
    quote do
      alias Nxgrad.{Activation, Neuron, Layer, MLP}
      import Nxgrad
    end
  end

  # API
  def run(model, xs, ys, rate) do
    case tuple_size(model) do
      3 ->
        {loss, grads} = Nx.Defn.value_and_grad(&run3(&1, xs, ys)).(model)
        {update_parameters3(model, grads, rate), to_number(loss)}

      4 ->
        {loss, grads} = Nx.Defn.value_and_grad(&run4(&1, xs, ys)).(model)
        {update_parameters4(model, grads, rate), to_number(loss)}
    end
  end

  defn run3({lin, ld1, lout}, xs, ys) do
    xs
    |> Layer.run(lin)
    |> Layer.run(ld1)
    |> Layer.run(lout)
    |> loss(ys)
  end

  defn run4({lin, ld1, ld2, lout}, xs, ys) do
    xs
    |> Layer.run(lin)
    |> Layer.run(ld1)
    |> Layer.run(ld2)
    |> Layer.run(lout)
    |> loss(ys)
  end

  defn loss(ypred, ys) do
    ypred
    |> Nx.subtract(ys)
    |> Nx.pow(2)
    |> Nx.sum()
  end

  defn update_parameters(values, grads, rate) do
    %Layer{values | b: values.b + grads.b * rate, ws: values.ws + grads.ws * rate}
  end

  defn update_parameters3({lin, ld1, lout}, {gin, gd1, gout}, rate) do
    lin = update_parameters(lin, gin, rate)
    ld1 = update_parameters(ld1, gd1, rate)
    lout = update_parameters(lout, gout, rate)

    {lin, ld1, lout}
  end

  defn update_parameters4({lin, ld1, ld2, lout}, {gin, gd1, gd2, gout}, rate) do
    lin = update_parameters(lin, gin, rate)
    ld1 = update_parameters(ld1, gd1, rate)
    ld2 = update_parameters(ld2, gd2, rate)
    lout = update_parameters(lout, gout, rate)

    {lin, ld1, ld2, lout}
  end

  def count_inactive(grads) do
    case tuple_size(grads) do
      3 -> count_inactive3(grads)
      4 -> count_inactive4(grads)
    end
  end

  # Behaviour

  def to_training_data(samples, batch_size) do
    type = :f64
    samples
    |> Stream.chunk_every(batch_size)
    |> Stream.map(fn batch ->
      {xs, ys} = Enum.unzip(batch)
      {Nx.tensor(xs, type: type), Nx.tensor(ys, type: type)}
    end)
  end

  def from_config(definition, opts) when length(definition) in [4, 5] do
    {:ok, MLP.from_config(definition, opts) |> List.to_tuple()}
  end

  def from_parameters(parameters) do
    {:ok, MLP.from_parameters(parameters) |> List.to_tuple()}
  end

  def to_parameters(model, _opts) do
    model
    |> Tuple.to_list()
    |> MLP.to_parameters()
  end

  def train({model, _loss}, data, rate) do
    Enum.reduce(data, {model, 0}, fn {xs, ys}, {model, _loss} ->
      xs = Nx.backend_copy(xs)
      ys = Nx.backend_copy(ys)

      run(model, xs, ys, rate)
    end)
  end

  # Private

  defp count_inactive3({gin, gd1, gout}) do
    iin = Nx.subtract(Nx.size(gin.b), Nx.any(gin.ws, axes: [1]) |> Nx.sum())
    id1 = Nx.subtract(Nx.size(gd1.b), Nx.any(gd1.ws, axes: [1]) |> Nx.sum())
    iout = Nx.subtract(Nx.size(gout.b), Nx.any(gout.ws, axes: [1]) |> Nx.sum())

    inactive = if iin > 0, do: [{:in, iin}], else: []

    inactive = if id1 > 0, do: [{:hidden, id1} | inactive], else: inactive

    if iout > 0, do: [{:out, iout} | inactive], else: inactive
  end

  defp count_inactive4({gin, gd1, gd2, gout}) do
    iin = nx_count_inactive(gin) |> Nx.to_number()
    id1 = nx_count_inactive(gd1) |> Nx.to_number()
    id2 = nx_count_inactive(gd2) |> Nx.to_number()
    iout = nx_count_inactive(gout) |> Nx.to_number()

    inactive = if iin > 0, do: [{:in, iin}], else: []

    inactive = if id1 > 0, do: [{:hidden1, id1} | inactive], else: inactive

    inactive = if id2 > 0, do: [{:hidden2, id2} | inactive], else: inactive

    if iout > 0, do: [{:out, iout} | inactive], else: inactive
  end

  defnp nx_count_inactive(grads) do
    Nx.sum(Nx.size(grads.b) - Nx.any(grads.ws, axes: [1]))
  end

  defp to_number(t) do
    cond do
      Nx.to_number(Nx.is_nan(t)) == 1 -> :NaN
      Nx.to_number(Nx.is_infinity(t)) == 1 -> :infinity
      true -> Nx.to_number(t)
    end
  end
end
