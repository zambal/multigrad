defmodule Multigrad.Activation do
  @type t :: :linear | :relu
  @type afun :: (number -> number)

  @spec afun(t) :: afun
  def afun(:linear), do: & &1
  def afun(:relu), do: &max(0, &1)
end
