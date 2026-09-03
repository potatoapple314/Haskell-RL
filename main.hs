-- {-# ... #-} は「プラグマ」。コンパイラへの指示で、コードではない。
-- LANGUAGE で言語拡張を有効化する。RankNTypes は型の中に forall を書けるようにする拡張
-- （lens の内部で ∀f. ... という多相型を使うために必要）。
{-# LANGUAGE RankNTypes #-}

-- module 名 where : このファイルが定義するモジュールの宣言。
-- 実行ファイルの入口は必ず「Main」モジュールの「main」関数になる約束。
module Main where

-- import : 他モジュールの定義を取り込む。Haskell では関数もこうして持ち込む。
--   import M            → M の公開名を全部使える（前置き不要）
--   import M (a, b)     → a, b だけ取り込む
--   import qualified M as N → N.xxx の形でしか使えない（名前衝突を防ぐ）
--   import M (T)        → 型 T を取り込む（Map は型名）
-- 本家 lens パッケージ(https://hackage.haskell.org/package/lens)を使用。
-- at / non / (^.) / (.~) / (%~) / (?~) / (&) はすべて Control.Lens 由来。
import Control.Lens                          -- レンズ関連を丸ごと取り込む
import qualified Data.Map.Strict as Map      -- 連想配列 Map。以後 Map.lookup のように使う
import Data.Map.Strict (Map)                 -- 型名 Map だけは前置きなしで使えるように
import Data.List (maximumBy)                 -- リストの最大要素を選ぶ関数
import Data.Ord (comparing)                  -- 比較関数を作るヘルパ
import System.Random (StdGen, mkStdGen, randomR)  -- 乱数生成器とその操作
import Text.Printf (printf)                  -- C言語風の書式付き出力

--------------------------------------------------
-- 強化学習用の一般化レンズ(パラメータ付きレンズ)
--------------------------------------------------

-- data 型名 型変数... = ... : 新しいデータ型の定義。
--   p x y e は「型変数」（小文字＝何の型でも入る穴。ジェネリクスのようなもの）。
--   = の右の Learner は「値を作るための構築子」（型名と同じ名前にするのが慣習）。
--   { フィールド名 :: 型, ... } は「レコード構文」。フィールド名は同時に
--   「その中身を取り出す関数」にもなる（例: forward someLearner :: p -> x -> y）。
--   :: は「〜という型を持つ」の意味。 -> は関数の矢印で、p -> x -> y は
--   「p を受け取り、x を受け取り、y を返す関数」（カリー化：引数は1つずつ渡る）。
-- Learner p x y e : パラメータ p を持つ「学習可能な射」。
--   forward  : 予測 (パラメータ p と入力 x から出力 y)
--   backward : 更新 (経験 e を使って p を更新)

-- | @Learner@ — パラメータ @p@ を持つ「学習可能な射」を表すレコード型。
--
-- @forward@ で予測を行い、@backward@ で経験からパラメータを更新する。
--
-- 型引数:
--
-- * @p@ — パラメータ（例: Q テーブル）
-- * @x@ — 入力（例: 状態）
-- * @y@ — 出力（例: 行動）
-- * @e@ — 経験（例: 遷移）
data Learner p x y e = Learner
  { forward  :: p -> x -> y
  , backward :: p -> x -> e -> p
  }

--------------------------------------------------
-- 状態と行動
--------------------------------------------------

-- 「列挙型」の定義。| は「または」で、値の候補を並べる。
--   State の値は Start か Middle か Terminal のどれか1つ（引数なしの構築子）。
--   deriving (...) : 型クラスの実装を自動生成してもらう指示。
--     Eq  → == で等値比較できる    Ord → < などで大小比較できる（宣言順が大小）
--     Show → show で文字列にできる（print で表示可能に）

-- | @State@ — 環境の状態。@Start@／@Middle@／@Terminal@ のいずれか 1 つ。
data State
  = Start
  | Middle
  | Terminal
  deriving (Eq, Ord, Show)

-- | @Action@ — エージェントが取れる行動。左へ（@GoLeft@）か右へ（@GoRight@）。
data Action
  = GoLeft
  | GoRight
  deriving (Eq, Ord, Show)

-- 型注釈と定義は別の行に書ける。上段が型、下段が中身。
--   [Action] は「Action のリスト」型。[a, b] はリストのリテラル。

-- | @actions@ — 選択可能な全行動のリスト（@[GoLeft, GoRight]@）。
actions :: [Action]
actions = [GoLeft, GoRight]

--------------------------------------------------
-- Q関数(永続Mapで表現)
--------------------------------------------------

-- type : 型シノニム（別名）。新しい型は作らず、既存型に名前を付けるだけ。
--   Map (State, Action) Double は「(State, Action) をキー、Double を値にする Map」。
--   (State, Action) は「タプル」型（2つ組）。Double は倍精度小数。

-- | @QTable@ — 行動価値関数 @Q(s, a)@ の表。@(状態, 行動)@ をキーに実数値を持つ。
type QTable = Map (State, Action) Double

-- 引数のない定義（＝ただの値）。Map.empty は空の Map。

-- | @emptyQTable@ — 空の Q テーブル。すべてのキーが未登録（＝ 0 扱い）。
emptyQTable :: QTable
emptyQTable = Map.empty

------------------------------------------------------------
-- ここから lens の使い方の説明。
-- レンズ = 「大きな構造の中の一部分」を指し示す道具で、読み・書き・更新を
-- 統一的に扱える。使う演算子はどれも Control.Lens 由来:
--   s ^. l         … s の中の l 部分を「読む」（view）
--   l .~ x         … l 部分を x に「置き換える」関数を作る（set）
--   l %~ f         … l 部分に関数 f を「適用して更新」する関数を作る（over）
--   l ?~ x         … Maybe を焦点にして「Just x を入れる」（挿入）
--   x & f          … f x と同じ。「x を f に通す」逆向き適用（データを左から右へ流す）
--   f . g          … 関数合成。(f . g) x = f (g x)。レンズも . で連結できる
------------------------------------------------------------

-- | @qEntry@ — Q テーブルの 1 マス @Q(state, action)@ を指すレンズを作る。
--
-- @at (state, action)@ で該当キーを @Maybe Double@ として覗き、@non 0@ で
-- 「未登録は 0 とみなし、0 の書き込みはキー削除にする」（疎表現の維持）。
-- @.@ で 2 つのレンズを繋ぎ、@QTable → Maybe Double → Double@ と焦点を絞る。
--
-- 引数:
--
-- * @state@ — 状態
-- * @action@ — 行動
qEntry :: State -> Action -> Lens' QTable Double
qEntry state action = at (state, action) . non 0

-- | @qValue@ — Q テーブルから @Q(state, action)@ を読み出す。
--
-- @q ^. qEntry state action@ は「@q@ にレンズ @qEntry state action@ を当てて読む」。
-- 未登録のキーは @qEntry@ 内の @non 0@ により 0 が返る。
--
-- 引数:
--
-- * @q@ — Q テーブル
-- * @state@ — 状態
-- * @action@ — 行動
qValue :: QTable -> State -> Action -> Double
qValue q state action = q ^. qEntry state action

--------------------------------------------------
-- 経験
--------------------------------------------------

-- レコード型（複数の値をまとめた「構造体」）。
--   フィールド名 transitionAction などは、取り出し関数としても使える。
--   例: reward t で t の reward を取得。Bool は真偽値（True / False）。

-- | @Transition@ — 環境から返る 1 回分の経験（遷移）。
--
-- フィールド:
--
-- * @transitionAction@ — 実際に取った行動
-- * @reward@ — 即時報酬
-- * @nextState@ — 遷移先の状態
-- * @finished@ — エピソードが終了したか
data Transition = Transition
  { transitionAction :: Action
  , reward           :: Double
  , nextState        :: State
  , finished         :: Bool
  }
  deriving Show

--------------------------------------------------
-- 方策
--------------------------------------------------

-- | @greedyAction@ — 貪欲方策。指定した状態で Q 値が最大の行動を選ぶ。
--
-- @comparing (qValue q state)@ で「行動を Q 値の大小で比べる」比較関数を作り、
-- @maximumBy@ でその比較における最大の行動（＝最良の行動）を返す。
--
-- 引数:
--
-- * @q@ — Q テーブル
-- * @state@ — 現在の状態
greedyAction :: QTable -> State -> Action
greedyAction q state =
  maximumBy
    (comparing (qValue q state))
    actions

--------------------------------------------------
-- Q-learning更新(永続Mapをレンズで更新)
--------------------------------------------------

-- | @updateQ@ — Q 学習の更新則で Q テーブルを 1 回更新する。
--
-- 更新式は @Q(s,a) ← Q(s,a) + α · (r + γ · max_{a'} Q(s',a') − Q(s,a))@。
-- 該当キーだけを新しい値に差し替えた「新しい Q テーブル」を返す（元は不変）。
--
-- 引数:
--
-- * @alpha@ — 学習率 α（0〜1、目標へどれだけ近づけるか）
-- * @gamma@ — 割引率 γ（0〜1、将来報酬の重み）
-- * @q@ — 現在の Q テーブル
-- * @state@ — 更新対象の状態
-- * @transition@ — 環境から得た経験（行動・報酬・次状態・終了フラグ）
--
-- 型が長いときは -> ごとに改行してよい。引数を順に受け、最後の QTable を返す。
updateQ
  :: Double
  -> Double
  -> QTable
  -> State
  -> Transition
  -> QTable
updateQ alpha gamma q state transition =
  -- q & レンズ操作 : q を左から右へ流し、レンズで更新した新しい QTable を得る。
  -- at (s,a) を必ず Just newValue にして挿入する(?~)。
  q & at (state, action) ?~ newValue
  -- where : この関数の中だけで使うローカル定義をまとめる場所（下で使う名前を後置きで定義）。
  --   評価は遅延なので、定義順は自由（使う場所より下に書ける）。
  where
    -- action : このレコードのフィールドを取り出す（transitionAction transition）。
    action =
      transitionAction transition

    oldValue =
      qValue q state action

    -- ガード（| 条件 = 値）。上から条件を見て、最初に真になった枝の値になる。
    --   otherwise は常に True の別名（＝「それ以外」の枝）。
    futureValue
      | finished transition = 0
      | otherwise =
          -- 「リスト内包表記」。 [ 式 | 変数 <- リスト ] で
          -- リストの各要素について式を計算した新リストを作る（数学の集合表記に似せた記法）。
          -- ここでは全 nextAction について Q 値を集め、maximum で最大を取る。
          maximum
            [ qValue q (nextState transition) nextAction | nextAction <- actions ]

    -- + * - は普通の中置演算子。字下げを続ければ式は複数行に書ける。
    target =
      reward transition
        + gamma * futureValue

    tdError =
      target - oldValue

    newValue =
      oldValue + alpha * tdError

--------------------------------------------------
-- Q-learning Learner
--------------------------------------------------

-- | @qLearningLearner@ — Q 学習を @Learner@ 型に梱包して返す。
--
-- @forward@ に貪欲方策 @greedyAction@ を、@backward@ に @updateQ alpha gamma@
-- （α, γ を部分適用したもの）を入れる。
--
-- 引数:
--
-- * @alpha@ — 学習率 α
-- * @gamma@ — 割引率 γ
qLearningLearner
  :: Double
  -> Double
  -> Learner QTable State Action Transition
qLearningLearner alpha gamma =
  Learner
    { forward = greedyAction
    , backward = updateQ alpha gamma
    }

--------------------------------------------------
-- 簡単な環境
--------------------------------------------------

-- | @environment@ — 環境の遷移関数。状態と行動から経験（@Transition@）を返す。
--
-- パターンマッチで @(状態, 行動)@ の組ごとに結果を定義している。
-- @Terminal@ は吸収状態で、どんな行動でも報酬 0 のまま終了し続ける。
--
-- 引数:
--
-- * 第1引数 — 現在の状態
-- * 第2引数 — 取る行動
--
-- 「パターンマッチ」で関数を定義。引数の形ごとに別々の式（＝節）を並べられる。
--   environment Start GoRight = ... のように、引数が特定の値のときの結果を書く。
--   上から順に照合し、最初に一致した節が使われる。全パターンを尽くすのが基本。
environment :: State -> Action -> Transition
environment Start GoRight =
  -- Transition { ... } でレコード値を生成。各フィールドに具体値を入れる。
  Transition
    { transitionAction = GoRight
    , reward = 0
    , nextState = Middle
    , finished = False
    }

environment Start GoLeft =
  Transition
    { transitionAction = GoLeft
    , reward = -1
    , nextState = Terminal
    , finished = True
    }

environment Middle GoRight =
  Transition
    { transitionAction = GoRight
    , reward = 1
    , nextState = Terminal
    , finished = True
    }

environment Middle GoLeft =
  Transition
    { transitionAction = GoLeft
    , reward = 0
    , nextState = Start
    , finished = False
    }

-- ここは第2引数を小文字 action にしている＝「どんな Action でも受ける変数パターン」。
--   具体値でなく変数を書くと、その位置は何にでもマッチしその値を束縛する。
environment Terminal action =
  Transition
    { transitionAction = action
    , reward = 0
    , nextState = Terminal
    , finished = True
    }

--------------------------------------------------
-- 1ステップ学習
--------------------------------------------------

-- | @trainStep@ — 貪欲方策で 1 ステップ学習を進める。
--
-- @forward@ で行動を選び、@environment@ で経験を得て、@backward@ で Q を更新する。
-- 更新後の Q テーブルと次状態の組を返す。
--
-- 引数:
--
-- * @learner@ — 使用する学習器
-- * @q@ — 現在の Q テーブル
-- * @state@ — 現在の状態
--
-- 戻り値 (QTable, State) はタプル（2つの値の組）。複数の結果をまとめて返せる。
trainStep
  :: Learner QTable State Action Transition
  -> QTable
  -> State
  -> (QTable, State)
trainStep learner q state =
  -- let 定義... in 式 : 局所的に名前を定義してから式を評価する構文。
  --   where と似るが、let は式の一部として使える（in の後の式で使う）。
  --   forward learner はレコードから forward 関数を取り出して呼んでいる。
  let
    action =
      forward learner q state

    transition =
      environment state action

    newQ =
      backward learner q state transition
  in
    -- (a, b) の形でタプルを作って返す。
    (newQ, nextState transition)

--------------------------------------------------
-- 指定された行動で学習
--
-- 最初はQ値が全て0なので、greedyだけでは探索できない。
-- 動作確認のため、行動を外から指定できるようにする。
--------------------------------------------------

-- | @trainWithAction@ — 行動を外から指定して 1 ステップ学習する。
--
-- @trainStep@ と違い方策で行動を選ばず、引数 @action@ をそのまま使う。
-- 学習初期のように Q 値が全て 0 で貪欲方策が機能しないときの動作確認に使う。
--
-- 引数:
--
-- * @learner@ — 使用する学習器
-- * @q@ — 現在の Q テーブル
-- * @state@ — 現在の状態
-- * @action@ — 実行する行動
trainWithAction
  :: Learner QTable State Action Transition
  -> QTable
  -> State
  -> Action
  -> (QTable, State)
-- trainStep とほぼ同じだが、action を選ばず引数で受け取る点だけ違う。
trainWithAction learner q state action =
  let
    transition =
      environment state action

    newQ =
      backward learner q state transition
  in
    (newQ, nextState transition)

--------------------------------------------------
-- ε-greedy方策(探索付き)
--
-- greedyだけだと同じ道しか通らず、全ての(状態,行動)を
-- 学習できない。確率εでランダムな行動を選び、探索させる。
--------------------------------------------------

-- | @epsilonGreedy@ — ε-greedy 方策。確率 ε でランダム、それ以外は貪欲に行動を選ぶ。
--
-- 乱数を 1 つ引き、ε 未満ならランダムな行動、そうでなければ @greedyAction@ を選ぶ。
-- 純粋関数なので、選んだ行動と「次に使う乱数生成器」の組を返す。
--
-- 引数:
--
-- * @epsilon@ — 探索率 ε（0〜1）
-- * @q@ — 現在の Q テーブル
-- * @state@ — 現在の状態
-- * @g@ — 乱数生成器
--
-- 乱数は「純粋」なので、生成器 g を渡すと (値, 新しいg) が返る。次は新しい g を使う。
epsilonGreedy :: Double -> QTable -> State -> StdGen -> (Action, StdGen)
epsilonGreedy epsilon q state g =
  -- let で「タプルを分解して」束縛できる。(roll, g1) = ... は
  --   右辺の1つ目を roll に、2つ目を g1 に入れる（パターン束縛）。
  --   (0.0, 1.0 :: Double) の :: Double は「この数は Double」という型注釈（曖昧さ回避）。
  let (roll, g1) = randomR (0.0, 1.0 :: Double) g
  -- if 条件 then A else B : 条件式。Haskell では else が必須（式なので必ず値を返す）。
  in if roll < epsilon
       then let (i, g2) = randomR (0, length actions - 1) g1  -- 0〜(個数-1)の整数
            in (actions !! i, g2)   -- xs !! i はリスト xs の i 番目の要素（0始まり）
       else (greedyAction q state, g1)

--------------------------------------------------
-- 1エピソード(Startから終端まで)を学習
--------------------------------------------------

-- | @runEpisode@ — 1 エピソード（@Start@ から終端まで）を学習する。
--
-- @where@ の局所関数 @go@ が再帰でループし、終端に着くか 100 ステップで停止する。
-- 各ステップで @epsilonGreedy@ が行動を選び、@trainWithAction@ が Q を更新する。
-- 最終的な Q テーブルと乱数生成器の組を返す。
--
-- 引数:
--
-- * @epsilon@ — 探索率 ε
-- * @learner@ — 使用する学習器
-- * @q@ — 開始時の Q テーブル
-- * @g@ — 乱数生成器
--
-- 引数の一部だけ書いて定義する例（ポイントフリー気味）。
--   runEpisode epsilon learner = go 0 Start は、残り2引数 (q, g) を go に任せる形。
--   go は下の where で定義した「ループ役」の局所関数。Haskell に for 文は無く、
--   再帰（自分自身を呼ぶ）で繰り返しを表現する。
runEpisode
  :: Double            -- ε(探索率)。-- で行コメント、型の途中にも書ける
  -> Learner QTable State Action Transition
  -> QTable
  -> StdGen
  -> (QTable, StdGen)
runEpisode epsilon learner = go (0 :: Int) Start
  where
    -- go は steps, state, q, g の4引数を取る。ガードで終了条件を判定。
    go steps state q g
      | state == Terminal = (q, g)   -- 終端に着いたら現状を返して終了
      | steps >= 100      = (q, g)   -- 無限ループ防止の安全弁
      | otherwise =
          -- let で複数のタプル束縛を並べて書ける（縦を揃えるのは見やすさのため）。
          let (action, g') = epsilonGreedy epsilon q state g
              (q', next)    = trainWithAction learner q state action
          in go (steps + 1) next q' g'   -- 更新した値で自分を呼び直す（＝次の一歩）

--------------------------------------------------
-- エピソードをn回繰り返す
--------------------------------------------------

-- | @trainEpisodes@ — エピソードを @n@ 回繰り返して学習する。
--
-- @where@ の @go@ が残り回数を数えながら @runEpisode@ を繰り返し呼ぶ。
-- 全エピソード終了後の Q テーブルと乱数生成器の組を返す。
--
-- 引数:
--
-- * @n@ — 繰り返すエピソード数
-- * @epsilon@ — 探索率 ε
-- * @learner@ — 使用する学習器
-- * @q@ — 開始時の Q テーブル
-- * @g@ — 乱数生成器
trainEpisodes
  :: Int
  -> Double
  -> Learner QTable State Action Transition
  -> QTable
  -> StdGen
  -> (QTable, StdGen)
-- n 回エピソードを回す。go は「残り回数 k」を数えながら再帰するカウンタ。
trainEpisodes n epsilon learner = go n
  where
    -- go の定義が2節ある。数値のパターンマッチ:
    --   go 0 ... → 残り0回なら終了（基底ケース）
    --   go k ... → それ以外は1回実行して k-1 で再帰
    go 0 q g = (q, g)
    go k q g =
      let (q', g') = runEpisode epsilon learner q g
      in go (k - 1) q' g'

--------------------------------------------------
-- Q表を見やすく表示
--------------------------------------------------

-- | @printQ@ — Q テーブルの各 @Q(s, a)@ を整形して標準出力に表示する。
--
-- @Start@ と @Middle@ について全行動の Q 値を @printf@ で 1 行ずつ出す
-- （@Terminal@ は表示対象外）。表示だけを行い意味ある値は返さない（@IO ()@）。
--
-- 引数:
--
-- * @q@ — 表示する Q テーブル
--
-- IO () : 「副作用（画面出力など）を行い、意味ある値は返さない」型。
--   () は「ユニット」（中身なしの値）。純粋関数と違い、IO は外界に触れる印。
printQ :: QTable -> IO ()
printQ q =
  -- mapM_ 動作 リスト : リストの各要素に IO 動作を順に実行する（結果は捨てる）。
  --   ここでは (s,a) の全組み合わせ（2重のリスト内包表記）に line を適用。
  mapM_ line [ (s, a) | s <- [Start, Middle], a <- actions ]
  where
    -- 引数側でタプルを分解: line (s, a) は組を受けて s と a に分ける。
    line (s, a) =
      -- printf は書式文字列に従って出力。%s=文字列, %f=小数, %-6s=左詰め幅6 など。
      --   show は値を文字列へ変換（deriving Show のおかげ）。\n は改行。
      printf "  Q(%-6s, %-7s) = % .4f\n" (show s) (show a) (qValue q s a)

--------------------------------------------------
-- 実行例
--------------------------------------------------

-- | @main@ — プログラムの入口。学習の収束のようすを標準出力に表示する。
--
-- α=0.1, γ=0.9, ε=0.2 の Q 学習器で、1・10・100・1000・10000 エピソード時点の
-- Q テーブルを順に表示し、最後に理論上の最適値 @Q*@ を参考として表示する。
--
-- main : プログラムの入口。型は IO ()（副作用を実行する動作）。
main :: IO ()
-- do : IO 動作を「上から順に」並べて実行するブロック（手続き的に書ける記法）。
main = do
  -- do の中の let は、以降で使う純粋な値をまとめて定義（in は不要）。
  let
    learner = qLearningLearner 0.1 0.9   -- α=0.1(学習率), γ=0.9(割引率)
    epsilon = 0.2                         -- 20%はランダム行動で探索
    gen0    = mkStdGen 42                 -- 乱数の種を固定(再現可能)

    -- 各チェックポイントまでに学習した累計エピソード数
    checkpoints = [1, 10, 100, 1000, 10000]

    -- step は (現在のQ, 乱数, 前回累計) と 目標累計 n を受け、追加学習して新しい3つ組を返す。
    -- 引数側でタプル (q, g, prev) を分解して受け取っている。
    step (q, g, prev) n =
      let (q', g') = trainEpisodes (n - prev) epsilon learner q g
      in (q', g', n)

    -- scanl 関数 初期値 リスト : 畳み込みの途中経過を全部集めたリストを返す。
    --   初期値から step を順に適用し、各段階の (Q, g, 累計) を並べる。
    results =
      scanl step (emptyQTable, gen0, 0) checkpoints

  -- putStrLn 文字列 : 文字列を改行付きで出力する IO 動作。
  putStrLn "収束のようす(エピソード数ごとのQ値):\n"
  -- mapM_ (\引数 -> do ...) リスト : 各要素に対して do ブロックを実行。
  --   \x -> ... は「ラムダ（無名関数）」。ここでは入れ子タプルを直接分解している。
  --   _ は「使わない値」を捨てるワイルドカード。
  mapM_
    (\((q, _, n), _) -> do
        printf "== %d エピソード後 ==\n" (n :: Int)   -- n :: Int で型を明示
        printQ q
        putStrLn "")
    -- zip a b : 2リストを組にして [(a0,b0),(a1,b1),...]。drop 1 は先頭要素を捨てる
    --   （scanl の初期値ぶんをスキップする）。
    (zip (drop 1 results) checkpoints)

  putStrLn "理論上の最適値 Q* (参考):"
  putStrLn "  Q(Start , GoRight) =  0.9000"
  putStrLn "  Q(Start , GoLeft ) = -1.0000"
  putStrLn "  Q(Middle, GoRight) =  1.0000"
  putStrLn "  Q(Middle, GoLeft ) =  0.8100"
