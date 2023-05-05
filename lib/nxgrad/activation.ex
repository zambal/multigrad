defmodule Nxgrad.Activation do
  @type afun :: (Nx.tensor() -> Nx.tensor())

  @spec afun(atom) :: afun
  def afun(:linear), do: & &1
  def afun(:relu), do: &Nx.max(0, &1)
  def afun(:softmax), do: fn act -> Nx.exp(act) / Nx.sum(Nx.exp(act)) end
end
