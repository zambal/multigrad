_ = """

get_data = fn ->
  Db.all(from m in MediaItem, where: not is_nil(m.labels))
  |> Enum.filter(fn m -> length(m.labels) > 0 end)
  |> Enum.map(fn m ->
    {m.labels, m.type, m.plan || "premium", m.asset_format || "none", !!m.no_rev_share, !!m.quality_stamp}
  end)
end

env = Vec.l(64)
m = Vec.m(env)
{env, m, l} = Vec.t(m, env)

"""

defmodule Vec do
  use Nxgrad

  defstruct seed: 1234,
            nin: 3,
            nout: 32,
            height1: 4096,
            height2: 4096,
            chunk_size: 256,
            count: 1_000,
            rate: -0.000001,
            sample_size: 0,
            s2i: %{},
            i2s: %{},
            items: [],
            data: nil,
            groups: [],
            samples: [],
            predict_fun: nil

  def i2v(i, size, def \\ -1.0), do: i2v(i, size, 0, def)

  defp i2v(_i, s, s, _def), do: []
  defp i2v(i, s, i, def), do: [1.0 | i2v(i, s, i + 1, def)]
  defp i2v(i, s, n, def), do: [def | i2v(i, s, n + 1, def)]

  def is2v(i, size, def \\ -1.0), do: is2v(i, size, 0, def)

  defp is2v(_is, s, s, _def), do: []

  defp is2v(is, s, n, def) do
    x = if n in is, do: 0.5, else: def
    [x | i2v(is, s, n + 1, def)]
  end

  def v2i(v), do: v2i(v, 0)

  defp v2i([x | _xs], n) when x > 0.25, do: n
  defp v2i([], _n), do: -1

  def v2is(v), do: v2is(v, 0)

  defp v2is([x | xs], n) when x > 0.25, do: [n | v2is(xs, n + 1)]
  defp v2is([_x | xs], n), do: v2is(xs, n + 1)
  defp v2is([], _n), do: []

  def s2v(s, %Vec{s2i: s2i} = env), do: Map.fetch!(s2i, s) |> i2v(env.sample_size)

  def ss2v(ss, %Vec{s2i: s2i} = env) do
    ss |> Enum.map(&Map.fetch!(s2i, &1)) |> is2v(env.sample_size)
  end

  def v2s(v, %Vec{i2s: i2s}), do: Map.get(i2s, v2i(v))

  def v2ss(v, %Vec{i2s: i2s}), do: v |> v2is() |> Enum.map(&Map.get(i2s, &1))

  def i2prop(0), do: "ad_dashboard"
  def i2prop(1), do: "album_track"
  def i2prop(2), do: "audio"
  def i2prop(3), do: "book"
  def i2prop(4), do: "brand"
  def i2prop(5), do: "category"
  def i2prop(6), do: "episode"
  def i2prop(7), do: "external"
  def i2prop(8), do: "issue"
  def i2prop(9), do: "magazine"
  def i2prop(10), do: "music_album"
  def i2prop(11), do: "photo_package"
  def i2prop(12), do: "photo"
  def i2prop(13), do: "podcast_episode"
  def i2prop(14), do: "podcast"
  def i2prop(15), do: "product_category"
  def i2prop(16), do: "product"
  def i2prop(17), do: "season"
  def i2prop(18), do: "series"
  def i2prop(19), do: "video"
  def i2prop(20), do: "year"
  def i2prop(21), do: "ad_supported"
  def i2prop(22), do: "free"
  def i2prop(23), do: "premium"
  def i2prop(24), do: "print"
  def i2prop(25), do: "image"
  def i2prop(26), do: "external_video"
  def i2prop(27), do: "external_print"
  def i2prop(28), do: "external_audio"
  def i2prop(29), do: "none"
  def i2prop(30), do: "no_rev_share"
  def i2prop(31), do: "quality_stamp"

  def prop2i("ad_dashboard"), do: 0
  def prop2i("album_track"), do: 1
  def prop2i("audio"), do: 2
  def prop2i("book"), do: 3
  def prop2i("brand"), do: 4
  def prop2i("category"), do: 5
  def prop2i("episode"), do: 6
  def prop2i("external"), do: 7
  def prop2i("issue"), do: 8
  def prop2i("magazine"), do: 9
  def prop2i("music_album"), do: 10
  def prop2i("photo_package"), do: 11
  def prop2i("photo"), do: 12
  def prop2i("podcast_episode"), do: 13
  def prop2i("podcast"), do: 14
  def prop2i("product_category"), do: 15
  def prop2i("product"), do: 16
  def prop2i("season"), do: 17
  def prop2i("series"), do: 18
  def prop2i("video"), do: 19
  def prop2i("year"), do: 20
  def prop2i("ad_supported"), do: 21
  def prop2i("free"), do: 22
  def prop2i("premium"), do: 23
  def prop2i("print"), do: 24
  def prop2i("image"), do: 25
  def prop2i("external_video"), do: 26
  def prop2i("external_print"), do: 27
  def prop2i("external_audio"), do: 28
  def prop2i("none"), do: 29
  def prop2i("no_rev_share"), do: 30
  def prop2i("quality_stamp"), do: 31

  def v2prop(v), do: v |> v2i() |> i2prop()
  def prop2v(s), do: s |> prop2i() |> i2v(32, -0.5)

  def v2props(v), do: v |> v2is() |> Enum.map(&i2prop/1)
  def props2v(ss), do: ss |> Enum.map(&prop2i/1) |> is2v(32, -0.5)

  def load(opts) do
    chunk_size = Keyword.get(opts, :chunk_size)

    data = File.read!("d.term") |> :erlang.binary_to_term() |> Enum.take(50)

    list =
      data
      |> Enum.map(fn {labels, _, _, _, _, _} -> labels end)
      |> List.flatten()
      |> Enum.uniq()
      |> Enum.sort()

    sample_size = length(list)

    s2i = Enum.with_index(list, &{&1, &2}) |> Enum.into(%{})

    i2s = Enum.with_index(list, &{&2, &1}) |> Enum.into(%{})

    env = %Vec{
      s2i: s2i,
      i2s: i2s,
      sample_size: sample_size,
      data: data,
      chunk_size: chunk_size,
      nin: sample_size
    }

    {xs, ys} =
      for {labels, type, plan, asset, no_rev, stamp} <- data do
        props =
          case {no_rev, stamp} do
            {true, true} -> ["no_rev_share", "quality_stamp"]
            {true, false} -> ["no_rev_share"]
            {false, true} -> ["quality_stamp"]
            {false, false} -> []
          end

        {ss2v(labels, env), props2v([type, plan, asset | props])}
      end
      |> Enum.unzip()

    xs = Enum.chunk_every(xs, chunk_size) |> Enum.map(&Nx.tensor/1)
    ys = Enum.chunk_every(ys, chunk_size) |> Enum.map(&Nx.tensor/1)

    {:ok, Enum.zip(xs, ys)}
  end

  def m(env, opts \\ []) do
    height1 = Keyword.get(opts, :height1, env.height1)
    height2 = Keyword.get(opts, :height2, env.height2)
    seed = Keyword.get(opts, :seed, env.seed)

    Nxgrad.from_config([env.nin, height1, height2, {env.nout, :softmax}], seed: seed)
  end

  def t(m, env, opts \\ []) do
    count = Keyword.get(opts, :count, env.count)
    rate = Keyword.get(opts, :rate, env.rate)

    {f, m, l} = Nxgrad.train(m, env.samples, count: count, rate: rate)

    {%Vec{env | predict_fun: f}, m, l}
  end

  def q(label, env), do: qs([label], env)

  def qs(labels, env) do
    v = ss2v(labels, env)
    env.predict_fun.(v) |> v2props()
  end
end
