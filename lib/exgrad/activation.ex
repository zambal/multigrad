defmodule Exgrad.Activation do
  alias Exgrad.Expr

  @type afun :: (Expr.t() -> Expr.t())

  @spec afun(atom) :: afun
  def afun(:linear), do: & &1
  def afun(:relu), do: &Expr.relu/1
end
