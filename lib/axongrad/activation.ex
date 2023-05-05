defmodule Axongrad.Activation do
  @type afun :: (Axon.t() -> Axon.t())

  @spec afun(atom) :: afun
  def afun(:linear), do: & &1
  def afun(:relu), do: &Axon.relu/1
  def afun(:softmax), do: &Axon.softmax/1
end
