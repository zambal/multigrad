defmodule MultigradTest do
  use ExUnit.Case
  doctest Multigrad

  alias Exgrad.Expr

  setup_all :start_server

  defp samples, do: [
    {[1, 2, 3], [10]},
    {[2, 3, 4], [20]},
    {[3, 4, 5], [30]},
    {[4, 5, 6], [40]}
  ]

  defp definition, do: [3, 4, 4, 1]

  def start_server(_), do: (Multigrad.start_link(rounds: 3_000, epochs_per_stats: 1_000); :ok)

  test "Nxgrad Neural Network check" do
    Multigrad.reset()
    Multigrad.set_module(Nxgrad)
    Multigrad.load(fn _opts -> {:ok, samples()} end)
    Multigrad.build(definition())
    Multigrad.await_ready()
    Multigrad.train()
    Multigrad.await_done()

    f = Multigrad.get_predict_fun()
    {_model, l} = Multigrad.get_model()

    [a] = f.([1, 2, 3])
    [b] = f.([2, 3, 4])
    [c] = f.([3, 4, 5])
    [d] = f.([4, 5, 6])

    assert(tr(l) == 0.0)

    assert(tr(a) == 10.0)
    assert(tr(b) == 20.0)
    assert(tr(c) == 30.0)
    assert(tr(d) == 40.0)
  end

  test "Exgrad Neural Network check" do
    use Exgrad

    Multigrad.reset()
    Multigrad.set_module(Exgrad)
    Multigrad.load(fn _opts -> {:ok, samples()} end)
    Multigrad.build(definition())
    Multigrad.await_ready()
    Multigrad.train()
    Multigrad.await_done()

    f = Multigrad.get_predict_fun()
    {_model, l} = Multigrad.get_model()

    [a] = f.([1, 2, 3])
    [b] = f.([2, 3, 4])
    [c] = f.([3, 4, 5])
    [d] = f.([4, 5, 6])

    assert(fr(l) == 0.0)

    assert(fr(a) == 10.0)
    assert(fr(b) == 20.0)
    assert(fr(c) == 30.0)
    assert(fr(d) == 40.0)
  end

  test "Neural Network compare" do
    use Exgrad

    Multigrad.reset()
    Multigrad.set_module(Exgrad)
    Multigrad.load(fn _opts -> {:ok, samples()} end)
    Multigrad.build(definition())
    Multigrad.await_ready()

    {:ok, init_params} = Multigrad.get_parameters()
    {:ok, m2} = Nxgrad.from_parameters(init_params)

    Multigrad.train()
    Multigrad.await_done()

    {:ok, p1} = Multigrad.get_parameters()

    Multigrad.reset()
    Multigrad.set_module(Nxgrad)
    Multigrad.load(fn _opts -> {:ok, samples()} end)
    Multigrad.put_model(m2)
    Multigrad.await_ready()
    Multigrad.train()
    Multigrad.await_done()

    {:ok, p2} = Multigrad.get_parameters()

    assert round_params(p1, 6) == round_params(p2, 6)
  end


  # lifted from https://github.com/karpathy/micrograd#example-usage

  test "Expr functions check" do
    a = Expr.value(-4.0, "a")
    b = Expr.value(2.0, "b")
    c = Expr.add(a, b)
    d = Expr.add(Expr.mul(a, b), Expr.pow(b, 3))
    c = Expr.add(c, Expr.add(c, 1))
    c = Expr.add(c, Expr.add(Expr.add(1, c), Expr.neg(a)))
    d = Expr.add(d, Expr.add(Expr.mul(d, 2), Expr.relu(Expr.add(b, a))))
    d = Expr.add(d, Expr.add(Expr.mul(3, d), Expr.relu(Expr.sub(b, a))))
    e = Expr.sub(c, d)
    f = Expr.pow(e, 2)
    g = Expr.div(f, 2.0)
    g = Expr.add(g, Expr.div(10.0, f))

    g = Expr.run(g)

    value_g = Float.round(g.value, 4)
    grad_a = Expr.grad(g, "a")
    grad_b = Expr.grad(g, "b")

    assert(fr(value_g) == 24.7041)
    assert(fr(grad_a) == 138.8338)
    assert(fr(grad_b) == 645.5773)
  end

  test "Expr algebra + transform check" do
    g =
      (fn ->
         use Exgrad.Expr.Algebra

         a = value(-5.0, "a")
         b = value(2.0, "b")
         c = a + b
         d = a * b + b ** 3
         c = c + c + 1
         c = c + 1 + c + -a
         d = d + d * 2 + relu(b + a)
         d = d + 3 * d + relu(b - a)
         e = c - d
         f = e ** 2
         g = f / 2.0
         g + 10.0 / f
       end).()

    g = Expr.update(g, "a", fn x -> x + 1 end)

    value_g = Float.round(g.value, 4)
    grad_a = Expr.grad(g, "a") |> Float.round(4)
    grad_b = Expr.grad(g, "b") |> Float.round(4)

    assert(value_g == 24.7041)
    assert(grad_a == 138.8338)
    assert(grad_b == 645.5773)
  end

  defp fr(x), do: Float.round(x * 1.0, 4)
  defp tr(x) do
    cond do
      Nx.to_number(Nx.is_nan(x)) == 1 -> :NaN
      Nx.to_number(Nx.is_infinity(x)) == 1 -> :infinity
      true -> x |> Nx.to_number() |> then(& &1 * 1.0) |> Float.round(4)
    end
  end

  defp round_params({bs, ws, act}, precision) do
    {Enum.map(bs, &Float.round(&1, precision)), Enum.map(ws, fn w -> Enum.map(w, &Float.round(&1, precision)) end), act}
  end
  defp round_params([p | ps], precision) do
    [round_params(p, precision) | round_params(ps, precision)]
  end
  defp round_params([], _precision) do
    []
  end
end
