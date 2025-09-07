# SQL解析ツール

## 概要

このツールは、指定ディレクトリ内の `.sql` ファイルを再帰的に検出し、各ファイルについて CRUD 操作の抽出、複雑度・サイズなどのメトリクスを算出、可視化、Markdown レポートを自動生成します。設計はモジュール化されており、解析ロジックは `sql-analyzer.py` 内のクラス群で構成されています。

## 主な機能

- SQLファイルの再帰検索とエンコーディング検出（BOM / chardet）
- コメント除去・正規化・文分割（`sqlparse` を利用）
- CRUD 操作（CREATE/READ/UPDATE/DELETE）の抽出（正規表現＋sqlparse ベースの二段解析）
- サブクエリ数、JOIN数、WHERE 条件数、CASE/CTE/関数呼び出し数などの複雑度メトリクス算出
- ファイルごとのサイズスコア・複雑度スコア算出と 3×3 クラスタ分類
- 集計結果の CSV 出力（`sql_comprehensive_metrics.csv`）とテーブル対象一覧（`sql_table_targets.csv`）
- matplotlib / seaborn による可視化（クラスタ散布図・CRUD 分布図）
- Markdown（`sql_analysis_report.md`）による解析レポート自動生成

## 必要環境

- Python 3.7 以上
- 依存パッケージはリポジトリルートの `requirement.txt` に列挙されています（例: pandas, chardet, matplotlib, seaborn, numpy, sqlparse）。

インストール例:

```sh
pip install -r requirement.txt
```

## 実行方法

```sh
python sql-analyzer.py <SQLファイルディレクトリ> [-o 出力先ディレクトリ] [-v]
```

- `<SQLファイルディレクトリ>`: 解析対象の SQL ファイルが格納されたディレクトリ（再帰検索）
- `-o` / `--output`: レポート等の出力先ディレクトリ（デフォルト: `./output`）
- `-v` / `--verbose`: デバッグ出力を有効にする

例:

```sh
python sql-analyzer.py ./sql_samples -o ./report -v
```

追加の実行制御:

- 環境変数 `SQL_ANALYZER_MAX_WORKERS` により並列ワーカー数を上書きできます（デフォルトは CPU を考慮した安全な値）。

## 出力ファイル

デフォルトの出力ディレクトリ（`./output`）に以下が生成されます:

- `sql_comprehensive_metrics.csv` : 各ファイルのメトリクス一覧（CSV）
- `sql_table_targets.csv` : ファイルごと／テーブルごとの CRUD 対象一覧（CSV）
- `sql_complexity_cluster_plot.png` : サイズ×複雑度のクラスタ散布図（PNG）
- `sql_crud_distribution.png` : 全体の CRUD 操作分布（PNG）
- `sql_analysis_report.md` : Markdown 形式の解析レポート

## 実装上の注意 / 制限

- 対応 SQL 方言は限定的です。方言検出は heuristic（特徴語のカウント）に基づくため完全ではありません。
- 大きなファイル保護: 解析対象 SQL が 10MB を超える場合は一部を切り出して解析します（解析時間とメモリの保護）。
- エンコーディング検出は BOM 判定 → chardet の順で行います。読み込みに失敗したファイルはログに記録されます。
- SQL の解析には `sqlparse` を使用しますが、すべての構文を完全に扱えるわけではないため、正規表現ベースのフォールバック処理も組み合わせています。

## クラス構成（概略）

- `SQLFileReader` : ファイル検出・エンコーディング読み込み
- `SQLParser` : コメント除去・正規化・文分割
- `CRUDAnalyzer` : CRUD 抽出（sqlparse + regex）
- `ComplexityAnalyzer` : 複雑度メトリクス計算
- `ClusterAnalyzer` : サイズ/複雑度の閾値管理とクラスタ分類
- `ReportGenerator` : DataFrame 生成・可視化・Markdown レポート生成
- `SQLAnalysisManager` : 全体ワークフロー管理（解析実行 → CSV/レポート出力）

## 参考 / カスタマイズ

- しきい値やラベルは `ClusterAnalyzer` のコンストラクタやプロパティを変更することで調整できます。
- 大規模解析や特殊な SQL 方言を正確に扱いたい場合は `CRUDAnalyzer` / `ComplexityAnalyzer` の正規表現や sqlparse のパースロジックを拡張してください。

---

*問題や追加したい情報があれば issue を立ててください。*
