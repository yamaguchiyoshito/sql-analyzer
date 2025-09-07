#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQL解析ツール：モジュラー設計による分離可能なSQLファイル解析ツール

"""

import os
import re
import sys
import argparse
import math
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import chardet
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import sqlparse
import logging
import concurrent.futures


class CRUDOperation(Enum):
    """CRUD操作の種類を定義する列挙型"""
    CREATE = 'C'
    READ = 'R'
    UPDATE = 'U'
    DELETE = 'D'

@dataclass
class FileAnalysisResult:
    """ファイル解析結果を格納するデータクラス"""
    file_name: str
    file_path: str
    content: str
    crud_operations: Dict[str, Set[CRUDOperation]]
    complexity_metrics: Dict[str, Any]
    cluster_classification: str

@dataclass  
class ComplexityMetrics:
    """複雑度メトリクスを格納するデータクラス"""
    total_lines: int
    non_empty_lines: int
    total_chars: int
    chars_no_whitespace: int
    statement_count: int
    subquery_count: int
    join_count: int
    where_condition_count: int
    case_count: int
    cte_count: int
    function_count: int
    unique_table_count: int
    complexity_score: float
    size_score: float

class SQLFileReader:
    """SQLファイルの読み込みとエンコーディング処理を担当するクラス"""
    
    def __init__(self):
        self.encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'shift_jis', 'utf-16']
        self.failed_files = []
    
    def find_sql_files(self, directory_path: str) -> List[str]:
        """指定ディレクトリ内のSQLファイルを再帰的に検索する"""
        sql_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.sql'):
                    full_path = os.path.join(root, file)
                    sql_files.append(full_path)
        return sql_files
    
    def detect_encoding(self, file_path: str) -> Optional[str]:
        """ファイルのエンコーディングを自動検出する"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(4096)
            
            # BOM検出
            if raw_data.startswith(b'\xef\xbb\xbf'):
                return 'utf-8-sig'
            elif raw_data.startswith(b'\xff\xfe') or raw_data.startswith(b'\xfe\xff'):
                return 'utf-16'
            
            # 自動検出
            result = chardet.detect(raw_data)
            return result['encoding'] if result['encoding'] else 'utf-8'
        except Exception:
            return None
    
    def read_file_content(self, file_path: str) -> Optional[str]:
        """ファイル内容を適切なエンコーディングで読み込む"""
        # エンコーディング検出を試行
        detected_encoding = self.detect_encoding(file_path)
        if detected_encoding:
            try:
                with open(file_path, 'r', encoding=detected_encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                pass
        
        # 複数エンコーディングでの試行
        for encoding in self.encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logging.warning(f"ファイル {file_path} 読み込み中に予期しないエラー: {e}")
                break
        
        # 読み込み失敗を記録
        self.failed_files.append(file_path)
        return None

class SQLParser:
    """SQL文の構文解析と前処理を担当するクラス"""
    
    @staticmethod
    def remove_comments(sql_content: str) -> str:
        """SQLコンテンツからコメントを除去する"""
        # 単行コメント除去
        content = re.sub(r'--.*?(?:\n|$)', ' ', sql_content)
        # 複数行コメント除去
        content = re.sub(r'/\*.*?\*/', ' ', content, flags=re.DOTALL)
        return content
    
    @staticmethod
    def normalize_sql(sql_content: str) -> str:
        """SQL文を正規化する（改行を空白に変換など）"""
        return re.sub(r'[\r\n\t]+', ' ', sql_content)
    
    @staticmethod
    def split_statements(sql_content: str) -> List[str]:
        """SQL文を分割する。sqlparse を使い、失敗時はセミコロン分割へフォールバックする。"""
        try:
            raw_stmts = sqlparse.split(sql_content)
            return [stmt.strip() for stmt in raw_stmts if stmt and stmt.strip()]
        except Exception:
            # フォールバック: 単純なセミコロン分割
            return [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
    
    def preprocess_sql(self, sql_content: str) -> Tuple[str, str, List[str]]:
        """SQL文の前処理を実行し、各形式を返す"""
        content_no_comments = self.remove_comments(sql_content)
        normalized_content = self.normalize_sql(content_no_comments)
        statements = self.split_statements(sql_content)
        
        return content_no_comments, normalized_content, statements

class CRUDAnalyzer:
    """CRUD操作の識別と抽出を担当するクラス"""
    
    def __init__(self, parser: SQLParser):
        self.parser = parser
        
        # 正規表現パターンを事前コンパイルしてパフォーマンスを向上
        self.create_table_pattern = re.compile(
            r'create\s+(?:table|or\s+replace\s+table)\s+(?:if\s+not\s+exists\s+)?(?:(\w+|[\"\[\]`\']+)\.)?(["\[\]`\'\w]+)', 
            re.IGNORECASE
        )
        self.create_view_pattern = re.compile(
            r'create\s+(?:or\s+replace\s+)?view\s+(?:if\s+not\s+exists\s+)?(?:(\w+|[\"\[\]`\']+)\.)?(["\[\]`\'\w]+)', 
            re.IGNORECASE
        )
        self.select_from_pattern = re.compile(
            r'(?:select|with\s+\w+\s+as).*?from\s+(?:(\w+|[\"\[\]`\']+)\.)?(["\[\]`\'\w]+)', 
            re.IGNORECASE | re.DOTALL
        )
        self.join_pattern = re.compile(
            r'(?:inner|left|right|full|cross)?\s*join\s+(?:(\w+|[\"\[\]`\']+)\.)?(["\[\]`\'\w]+)',
            re.IGNORECASE
        )
        self.update_pattern = re.compile(
            r'update\s+(?:(\w+|[\"\[\]`\']+)\.)?(["\[\]`\'\w]+)', 
            re.IGNORECASE
        )
        self.delete_pattern = re.compile(
            r'delete\s+(?:from\s+)?(?:(\w+|[\"\[\]`\']+)\.)?(["\[\]`\'\w]+)', 
            re.IGNORECASE
        )
        self.insert_pattern = re.compile(
            r'insert\s+(?:into\s+)?(?:(\w+|[\"\[\]`\']+)\.)?(["\[\]`\'\w]+)', 
            re.IGNORECASE
        )
        self.merge_pattern = re.compile(
            r'merge\s+(?:into\s+)?(?:(\w+|[\"\[\]`\']+)\.)?(["\[\]`\'\w]+)', 
            re.IGNORECASE
        )
        self.truncate_pattern = re.compile(
            r'truncate\s+(?:table\s+)?(?:(\w+|[\"\[\]`\']+)\.)?(["\[\]`\'\w]+)', 
            re.IGNORECASE
        )
        
        # 無視すべきテーブル名のセット
        self.ignored_tables = {'dual', 'on', 'using', 'values', 'table', 'view', 'index', 'sequence'}
        
        # 最大解析サイズ (10MB)
        self.max_content_size = 10_000_000
    
    def extract_crud_operations(self, sql_content: str) -> Dict[str, Set[CRUDOperation]]:
        """SQLコンテンツからCRUD操作とテーブル名を抽出する"""
        try:
            # 入力サイズのチェック
            if not sql_content:
                return {}
                
            if len(sql_content) > self.max_content_size:
                logging.warning(f"SQL文が大きすぎます ({len(sql_content)}バイト)。部分解析を実行します。")
                sql_content = sql_content[:self.max_content_size]
            
            content_no_comments, normalized_content, statements = self.parser.preprocess_sql(sql_content)
            operations = defaultdict(set)

            # sqlparse を使った厳密解析（ステートメント単位）
            for stmt in statements:
                try:
                    stmt_strip = stmt.strip()
                    if not stmt_strip:
                        continue

                    stype = self._get_statement_type(stmt_strip)
                    tables = self._extract_tables_sqlparse(stmt_strip)

                    for table in tables:
                        if stype == 'select':
                            operations[table].add(CRUDOperation.READ)
                        elif stype == 'update':
                            operations[table].add(CRUDOperation.UPDATE)
                        elif stype == 'delete':
                            operations[table].add(CRUDOperation.DELETE)
                        elif stype == 'insert':
                            operations[table].add(CRUDOperation.CREATE)
                        elif stype == 'create':
                            operations[table].add(CRUDOperation.CREATE)
                        elif stype == 'merge':
                            operations[table].add(CRUDOperation.UPDATE)
                            operations[table].add(CRUDOperation.CREATE)
                        elif stype == 'truncate':
                            operations[table].add(CRUDOperation.DELETE)
                except Exception as e:
                    logging.debug(f"sqlparse ベースの解析でエラー: {e}")

            # regex ベースの検出をフォールバックとして残す（取りこぼし対策）
            sql_lower = normalized_content.lower()
            self._detect_create_operations(sql_lower, operations)
            self._detect_read_operations(sql_lower, operations)
            self._detect_update_operations(sql_lower, operations)
            self._detect_delete_operations(sql_lower, operations)
            self._detect_insert_operations(sql_lower, operations)
            self._detect_merge_operations(sql_lower, operations)
            self._detect_truncate_operations(sql_lower, operations)
            
            # 空のエントリやノイズを除去
            self._clean_operations(operations)
            
            return operations
        except Exception as e:
            logging.error(f"CRUD操作抽出中にエラーが発生しました: {e}")
            return {}
    
    def _clean_operations(self, operations: Dict[str, Set[CRUDOperation]]):
        """操作リストから無効なエントリを除去する"""
        # 削除対象のキーを特定
        keys_to_remove = []
        for table in operations.keys():
            # 空文字、数字のみ、予約語などを除外
            if (not table or table.isdigit() or 
                table.lower() in self.ignored_tables or
                len(table) <= 1):
                keys_to_remove.append(table)
        
        # 削除実行
        for key in keys_to_remove:
            del operations[key]
    
    def _clean_table_name(self, table: str) -> str:
        """テーブル名から不要な文字を除去する"""
        if not table:
            return ''
        # トリム
        t = table.strip()
        # 丸括弧や不要な前後記号を除去
        t = re.sub(r'^[\(\s]+|[\)\s]+$', '', t)
        # 引用符、角括弧、バッククォート、空白を除去（ドットは残す）
        t = re.sub(r'["\[\]`\'\s]', '', t)
        # ドットの連続を単一に
        t = re.sub(r'\.{2,}', '.', t)
        # 前後ドットを取り除く
        t = t.strip('.')
        # 小文字化して返す
        return t.lower()

    def _get_statement_type(self, statement: str) -> str:
        """ステートメントの先頭キーワードからタイプを推定する"""
        try:
            parsed = sqlparse.parse(statement)
            if not parsed:
                return ''
            stmt = parsed[0]
            # トークンを順に見て最初の DML / Keyword を検出する
            for token in stmt.flatten():
                try:
                    tok_val = str(token).strip()
                    if not tok_val:
                        continue
                    low = tok_val.lower()
                    # 直接的なキーワードで判断
                    if low.startswith('select') or low == 'with':
                        return 'select'
                    if low.startswith('update'):
                        return 'update'
                    if low.startswith('delete'):
                        return 'delete'
                    if low.startswith('insert'):
                        return 'insert'
                    if low.startswith('create'):
                        return 'create'
                    if low.startswith('merge'):
                        return 'merge'
                    if low.startswith('truncate'):
                        return 'truncate'
                except Exception:
                    continue
            return ''
        except Exception:
            return ''
    def _extract_tables_sqlparse(self, statement: str) -> Set[str]:
        """sqlparse を使ってステートメント内のテーブル識別子を抽出する"""
        tables = set()
        try:
            from sqlparse.sql import IdentifierList, Identifier
            parsed = sqlparse.parse(statement)
            if not parsed:
                return tables

            def extract_from_identifier(iden: 'Identifier') -> None:
                try:
                    # get_real_name はエイリアスを考慮した実名を返す
                    name = iden.get_real_name() or iden.get_name() or str(iden)
                    parent = iden.get_parent_name()
                    if parent:
                        full = f"{parent}.{name}"
                    else:
                        full = name
                    full = self._clean_table_name(full)
                    if full and full.lower() not in self.ignored_tables:
                        tables.add(full)
                except Exception:
                    return

            def recurse_tokens(token_list):
                for token in token_list:
                    # IdentifierList: 複数の識別子を含む
                    if isinstance(token, IdentifierList):
                        for iden in token.get_identifiers():
                            extract_from_identifier(iden)
                    # 単一の Identifier
                    elif isinstance(token, Identifier):
                        extract_from_identifier(token)
                    # グループトークン（括弧やサブクエリなど）は再帰処理
                    elif getattr(token, 'is_group', False):
                        recurse_tokens(getattr(token, 'tokens', []))
                    else:
                        # 直前キーワードが FROM/JOIN/INTO/UPDATE/DELETE/MERGE/USING/TABLE の場合、次の Name を候補として扱う
                        try:
                            val = str(token).strip()
                            if not val:
                                continue
                            low = val.lower()
                            # 一部キーワードは後続にテーブル名を期待
                            if low in ('from', 'join', 'into', 'update', 'delete', 'merge', 'using', 'table'):
                                # 次の有意トークンを探す
                                # token_list は親のリストなので現在の token の次を探す
                                # 実装上、親ループで次のトークンを処理するため、ここでは何もしない
                                pass
                        except Exception:
                            continue

            for stmt in parsed:
                recurse_tokens(stmt.tokens)

            # フィルタと正規化
            cleaned = set()
            for t in tables:
                t_clean = t.strip().lower()
                if not t_clean or t_clean.isdigit() or t_clean in self.ignored_tables or len(t_clean) <= 1:
                    continue
                cleaned.add(t)

            return cleaned
        except Exception as e:
            logging.debug(f"テーブル抽出で sqlparse エラー: {e}")
            return set()
    
    def _detect_create_operations(self, sql_lower: str, operations: Dict[str, Set[CRUDOperation]]):
        """CREATE操作を検出する"""
        try:
            # CREATE TABLE
            create_tables = self.create_table_pattern.findall(sql_lower)
            for schema, table in create_tables:
                table = self._clean_table_name(table)
                if table:
                    operations[table].add(CRUDOperation.CREATE)
            
            # CREATE VIEW
            create_views = self.create_view_pattern.findall(sql_lower)
            for schema, view in create_views:
                view = self._clean_table_name(view)
                if view:
                    operations[view].add(CRUDOperation.CREATE)
        except Exception as e:
            logging.error(f"CREATE操作検出中にエラーが発生しました: {e}")
    
    def _detect_read_operations(self, sql_lower: str, operations: Dict[str, Set[CRUDOperation]]):
        """READ操作を検出する"""
        try:
            # SELECT FROM
            read_tables = self.select_from_pattern.findall(sql_lower)
            for schema, table in read_tables:
                table = self._clean_table_name(table)
                if table and not table.isdigit() and table.lower() not in self.ignored_tables:
                    operations[table].add(CRUDOperation.READ)
            
            # JOIN操作
            join_tables = self.join_pattern.findall(sql_lower)
            for schema, table in join_tables:
                table = self._clean_table_name(table)
                if table and not table.isdigit() and table.lower() not in self.ignored_tables:
                    operations[table].add(CRUDOperation.READ)
        except Exception as e:
            logging.error(f"READ操作検出中にエラーが発生しました: {e}")
    
    def _detect_update_operations(self, sql_lower: str, operations: Dict[str, Set[CRUDOperation]]):
        """UPDATE操作を検出する"""
        try:
            update_tables = self.update_pattern.findall(sql_lower)
            for schema, table in update_tables:
                table = self._clean_table_name(table)
                if table:
                    operations[table].add(CRUDOperation.UPDATE)
        except Exception as e:
            logging.error(f"UPDATE操作検出中にエラーが発生しました: {e}")
    
    def _detect_delete_operations(self, sql_lower: str, operations: Dict[str, Set[CRUDOperation]]):
        """DELETE操作を検出する"""
        try:
            delete_tables = self.delete_pattern.findall(sql_lower)
            for schema, table in delete_tables:
                table = self._clean_table_name(table)
                if table:
                    operations[table].add(CRUDOperation.DELETE)
        except Exception as e:
            logging.error(f"DELETE操作検出中にエラーが発生しました: {e}")
    
    def _detect_insert_operations(self, sql_lower: str, operations: Dict[str, Set[CRUDOperation]]):
        """INSERT操作を検出する"""
        try:
            insert_tables = self.insert_pattern.findall(sql_lower)
            for schema, table in insert_tables:
                table = self._clean_table_name(table)
                if table:
                    operations[table].add(CRUDOperation.CREATE)  # INSERTはCREATEとして分類
        except Exception as e:
            logging.error(f"INSERT操作検出中にエラーが発生しました: {e}")
    
    def _detect_merge_operations(self, sql_lower: str, operations: Dict[str, Set[CRUDOperation]]):
        """MERGE/UPSERT操作を検出する"""
        try:
            merge_tables = self.merge_pattern.findall(sql_lower)
            for schema, table in merge_tables:
                table = self._clean_table_name(table)
                if table:
                    # MERGEはUPDATEとCREATE両方の操作として扱う
                    operations[table].add(CRUDOperation.UPDATE)
                    operations[table].add(CRUDOperation.CREATE)
        except Exception as e:
            logging.error(f"MERGE操作検出中にエラーが発生しました: {e}")
    
    def _detect_truncate_operations(self, sql_lower: str, operations: Dict[str, Set[CRUDOperation]]):
        """TRUNCATE操作を検出する"""
        try:
            truncate_tables = self.truncate_pattern.findall(sql_lower)
            for schema, table in truncate_tables:
                table = self._clean_table_name(table)
                if table:
                    # TRUNCATEはDELETEとして分類
                    operations[table].add(CRUDOperation.DELETE)
        except Exception as e:
            logging.error(f"TRUNCATE操作検出中にエラーが発生しました: {e}")
    
    def detect_sql_dialect(self, sql_content: str) -> str:
        """SQLの方言を検出する試みを行う"""
        try:
            # SQL方言の特徴的なキーワードやパターン
            dialect_patterns = {
                'Oracle': [r'\bconnect\s+by\b', r'\bdual\b', r'\bsysdate\b', r'\brownum\b', r'\bnvl\b'],
                'MySQL': [r'\blimit\b.*\boffset\b', r'\bshow\s+tables\b', r'\bnow\(\)', r'\bifnull\b'],
                'PostgreSQL': [r'\bnow\(\)::\b', r'\btimestamp\s+with\s+time\s+zone\b', r'\breturning\b'],
                'SQLServer': [r'\btop\s+\d+\b', r'\bnolock\b', r'\bgetdate\(\)', r'\bdeclare\s+@'],
                'SQLite': [r'\bsqlite_master\b', r'\bsqlite_sequence\b', r'\bdatetime\(\'now\'\)']
            }
            
            scores = {dialect: 0 for dialect in dialect_patterns}
            
            # 各方言の特徴的なパターンをカウント
            for dialect, patterns in dialect_patterns.items():
                for pattern in patterns:
                    matches = len(re.findall(pattern, sql_content, re.IGNORECASE))
                    scores[dialect] += matches
            
            # 最高スコアの方言を返す
            if max(scores.values()) > 0:
                return max(scores.items(), key=lambda x: x[1])[0]
            
            return "不明"
        except Exception as e:
            logging.error(f"SQL方言検出中にエラーが発生しました: {e}")
            return "不明"

class ComplexityAnalyzer:
    """複雑度メトリクスの計算を担当するクラス"""
    
    def __init__(self, parser: SQLParser):
        self.parser = parser
        # 正規表現パターンを事前コンパイルして再利用性を高める
        self.subquery_pattern = re.compile(r'\(\s*select\s+')
        self.case_pattern = re.compile(r'\bcase\s+')
        self.cte_pattern = re.compile(r'\bwith\s+\w+\s+as\s*\(')
        self.join_patterns = [
            re.compile(pattern) for pattern in [
                r'\s+join\s+', r'\s+inner\s+join\s+', r'\s+left\s+join\s+',
                r'\s+right\s+join\s+', r'\s+full\s+join\s+', r'\s+cross\s+join\s+'
            ]
        ]
        self.function_patterns = [
            re.compile(pattern) for pattern in [
                r'\b(?:count|sum|avg|max|min|coalesce|nvl|isnull|concat|substring|upper|lower|trim|cast|convert)\s*\(',
                r'\b(?:date|timestamp|now|current_date|current_timestamp)\s*\(',
                r'\b(?:row_number|rank|dense_rank|lag|lead|first_value|last_value)\s*\('
            ]
        ]
        self.table_patterns = [
            re.compile(pattern) for pattern in [
                r'from\s+(?:\w+\.)?(\w+)',
                r'join\s+(?:\w+\.)?(\w+)',
                r'update\s+(?:\w+\.)?(\w+)',
                r'insert\s+into\s+(?:\w+\.)?(\w+)',
                r'delete\s+from\s+(?:\w+\.)?(\w+)'
            ]
        ]
        self.where_pattern = re.compile(r'where\s+(.+?)(?=\s+(?:group\s+by|having|order\s+by|limit|;|$))', re.DOTALL)
        self.condition_split_pattern = re.compile(r'\s+(?:and|or)\s+')
    
    def calculate_complexity_metrics(self, sql_content: str) -> ComplexityMetrics:
        """SQLコンテンツから複雑度メトリクスを計算する"""
        try:
            if not sql_content or not sql_content.strip():
                # 空のSQLに対する安全な処理
                return ComplexityMetrics(
                    total_lines=0, non_empty_lines=0, total_chars=0, chars_no_whitespace=0,
                    statement_count=0, subquery_count=0, join_count=0, where_condition_count=0,
                    case_count=0, cte_count=0, function_count=0, unique_table_count=0,
                    complexity_score=0.0, size_score=0.0
                )
            
            content_no_comments, normalized_content, statements = self.parser.preprocess_sql(sql_content)
            
            # 基本メトリクス
            lines = sql_content.split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            
            basic_metrics = self._calculate_basic_metrics(sql_content, lines, non_empty_lines, statements)
            structural_metrics = self._calculate_structural_metrics(normalized_content.lower())
            
            # 複雑度・サイズスコアの計算
            complexity_score = self._calculate_complexity_score(structural_metrics)
            size_score = self._calculate_size_score(basic_metrics, len(statements))
            
            return ComplexityMetrics(
                **basic_metrics,
                **structural_metrics,
                complexity_score=complexity_score,
                size_score=size_score
            )
        except Exception as e:
            logging.error(f"複雑度分析中にエラーが発生しました: {e}")
            # 最小限のメトリクスを返す
            return ComplexityMetrics(
                total_lines=0, non_empty_lines=0, total_chars=0, chars_no_whitespace=0,
                statement_count=0, subquery_count=0, join_count=0, where_condition_count=0,
                case_count=0, cte_count=0, function_count=0, unique_table_count=0,
                complexity_score=0.0, size_score=0.0
            )
    
    def _calculate_basic_metrics(self, sql_content: str, lines: List[str], non_empty_lines: List[str], statements: List[str]) -> Dict[str, int]:
        """基本メトリクスを計算する"""
        try:
            return {
                'total_lines': len(lines),
                'non_empty_lines': len(non_empty_lines),
                'total_chars': len(sql_content),
                'chars_no_whitespace': len(re.sub(r'\s', '', sql_content)),
                'statement_count': len(statements)
            }
        except Exception as e:
            logging.error(f"基本メトリクス計算中にエラーが発生しました: {e}")
            return {
                'total_lines': 0, 'non_empty_lines': 0, 'total_chars': 0,
                'chars_no_whitespace': 0, 'statement_count': 0
            }
    
    def _calculate_structural_metrics(self, sql_lower: str) -> Dict[str, int]:
        """構造的複雑度メトリクスを計算する"""
        try:
            # リミットを設定して異常に大きな入力に対する防御
            if len(sql_lower) > 10_000_000:  # 10MB制限
                logging.warning(f"SQL文が大きすぎます ({len(sql_lower)}バイト)。部分解析を実行します。")
                sql_lower = sql_lower[:10_000_000]  # 10MBまでに制限
            
            # サブクエリ数
            subquery_count = len(self.subquery_pattern.findall(sql_lower))
            
            # JOIN数
            join_count = sum(len(pattern.findall(sql_lower)) for pattern in self.join_patterns)
            
            # WHERE条件数
            where_condition_count = self._count_where_conditions(sql_lower)
            
            # CASE文数
            case_count = len(self.case_pattern.findall(sql_lower))
            
            # CTE数
            cte_count = len(self.cte_pattern.findall(sql_lower))
            
            # 関数呼び出し数
            function_count = self._count_function_calls(sql_lower)
            
            # ユニークテーブル数
            unique_table_count = self._count_unique_tables(sql_lower)
            
            return {
                'subquery_count': subquery_count,
                'join_count': join_count,
                'where_condition_count': where_condition_count,
                'case_count': case_count,
                'cte_count': cte_count,
                'function_count': function_count,
                'unique_table_count': unique_table_count
            }
        except Exception as e:
            logging.error(f"構造メトリクス計算中にエラーが発生しました: {e}")
            return {
                'subquery_count': 0, 'join_count': 0, 'where_condition_count': 0,
                'case_count': 0, 'cte_count': 0, 'function_count': 0, 'unique_table_count': 0
            }
    
    def _count_where_conditions(self, sql_lower: str) -> int:
        """WHERE句の条件数を計算する"""
        try:
            where_conditions = 0
            where_matches = self.where_pattern.findall(sql_lower)
            for where_clause in where_matches:
                conditions = self.condition_split_pattern.split(where_clause)
                where_conditions += len(conditions)
            return where_conditions
        except Exception as e:
            logging.error(f"WHERE条件数計算中にエラーが発生しました: {e}")
            return 0
    
    def _count_function_calls(self, sql_lower: str) -> int:
        """関数呼び出し数を計算する"""
        try:
            return sum(len(pattern.findall(sql_lower)) for pattern in self.function_patterns)
        except Exception as e:
            logging.error(f"関数呼び出し数計算中にエラーが発生しました: {e}")
            return 0
    
    def _count_unique_tables(self, sql_lower: str) -> int:
        """ユニークテーブル数を計算する"""
        try:
            table_names = set()
            for pattern in self.table_patterns:
                matches = pattern.findall(sql_lower)
                # 無効な値をフィルタリング
                valid_matches = [m for m in matches if m and not m.isdigit() and m.lower() not in ('dual', 'on', 'using')]
                table_names.update(valid_matches)
            return len(table_names)
        except Exception as e:
            logging.error(f"ユニークテーブル数計算中にエラーが発生しました: {e}")
            return 0
    
    def _calculate_complexity_score(self, structural_metrics: Dict[str, int]) -> float:
        """複雑度スコアを計算する（重み付き合計）"""
        try:
            return (
                structural_metrics.get('subquery_count', 0) * 3 +
                structural_metrics.get('join_count', 0) * 2 +
                structural_metrics.get('where_condition_count', 0) * 1 +
                structural_metrics.get('case_count', 0) * 2 +
                structural_metrics.get('cte_count', 0) * 3 +
                structural_metrics.get('function_count', 0) * 1 +
                structural_metrics.get('unique_table_count', 0) * 1
            )
        except Exception as e:
            logging.error(f"複雑度スコア計算中にエラーが発生しました: {e}")
            return 0.0
    
    def _calculate_size_score(self, basic_metrics: Dict[str, int], statement_count: int) -> float:
        """サイズスコアを計算する"""
        try:
            return basic_metrics.get('non_empty_lines', 0) * 1 + max(0, statement_count) * 5
        except Exception as e:
            logging.error(f"サイズスコア計算中にエラーが発生しました: {e}")
            return 0.0

class ClusterAnalyzer:
    """クラスタリングと分類を担当するクラス"""
    
    def __init__(self, size_thresholds: Tuple[float, float] = None, 
                complexity_thresholds: Tuple[float, float] = None,
                auto_adjust: bool = False):
        # デフォルトのしきい値設定
        self.size_thresholds = size_thresholds or (50, 150)  # 小≤50 < 中≤150 < 大
        self.complexity_thresholds = complexity_thresholds or (10, 20)  # 低≤10 < 中≤20 < 高
        self.auto_adjust = auto_adjust
        
        # 言語ローカライズマッピング
        self.size_class_labels = {
            'ja': ['小', '中', '大'],
            'en': ['Small', 'Medium', 'Large']
        }
        
        self.complexity_class_labels = {
            'ja': ['低', '中', '高'],
            'en': ['Low', 'Medium', 'High']
        }
        
        # クラスタの統計情報
        self.cluster_stats = defaultdict(int)
        self.total_files = 0
        
        # 現在の言語設定
        self.current_language = 'ja'
    
    def set_language(self, language_code: str) -> None:
        """言語設定を変更する"""
        if language_code in self.size_class_labels:
            self.current_language = language_code
        else:
            logging.warning(f"サポートされていない言語コード '{language_code}'。デフォルト値を使用します。")
    
    def adjust_thresholds(self, file_metrics: List[Dict[str, float]]) -> None:
        """データに基づいて自動的にしきい値を調整する"""
        if not file_metrics or not self.auto_adjust:
            return
            
        try:
            # サイズとスコアの値を抽出
            size_scores = [m.get('size_score', 0) for m in file_metrics if 'size_score' in m]
            complexity_scores = [m.get('complexity_score', 0) for m in file_metrics if 'complexity_score' in m]
            
            if not size_scores or not complexity_scores:
                return
                
            # パーセンタイルベースのしきい値計算
            size_scores.sort()
            complexity_scores.sort()
            
            # 33パーセンタイルと66パーセンタイルを使用
            size_p33 = size_scores[int(len(size_scores) * 0.33)]
            size_p66 = size_scores[int(len(size_scores) * 0.66)]
            
            complexity_p33 = complexity_scores[int(len(complexity_scores) * 0.33)]
            complexity_p66 = complexity_scores[int(len(complexity_scores) * 0.66)]
            
            # しきい値更新（値が合理的な範囲内の場合）
            if 5 <= size_p33 <= 100 and 50 <= size_p66 <= 300:
                self.size_thresholds = (size_p33, size_p66)
            
            if 3 <= complexity_p33 <= 15 and 10 <= complexity_p66 <= 40:
                self.complexity_thresholds = (complexity_p33, complexity_p66)
                
                logging.info(f"しきい値を自動調整: サイズ={self.size_thresholds}, 複雑度={self.complexity_thresholds}")
        except Exception as e:
            logging.error(f"しきい値調整中にエラーが発生しました: {e}")
    
    def classify_into_cluster(self, size_score: float, complexity_score: float) -> str:
        """サイズと複雑度スコアに基づいてクラスタを分類する"""
        try:
            # サイズ分類
            if size_score <= self.size_thresholds[0]:
                size_idx = 0  # 小
            elif size_score <= self.size_thresholds[1]:
                size_idx = 1  # 中
            else:
                size_idx = 2  # 大
            
            # 複雑度分類
            if complexity_score <= self.complexity_thresholds[0]:
                complexity_idx = 0  # 低
            elif complexity_score <= self.complexity_thresholds[1]:
                complexity_idx = 1  # 中
            else:
                complexity_idx = 2  # 高
            
            size_class = self.size_class_labels[self.current_language][size_idx]
            complexity_class = self.complexity_class_labels[self.current_language][complexity_idx]
            
            # クラスタ名の生成
            cluster_name = f"{size_class}サイズ・{complexity_class}複雑度"
            
            # 統計更新
            self.cluster_stats[cluster_name] += 1
            self.total_files += 1
            
            return cluster_name
        except Exception as e:
            logging.error(f"クラスタ分類中にエラーが発生しました: {e}")
            return "分類不能"
    
    def get_cluster_thresholds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """クラスタ分類のしきい値を取得する"""
        return self.size_thresholds, self.complexity_thresholds
    
    def get_cluster_distribution(self) -> Dict[str, float]:
        """クラスタの分布率を計算する"""
        if self.total_files == 0:
            return {}
        
        return {cluster: (count / self.total_files) * 100 for cluster, count in self.cluster_stats.items()}
    
    def get_cluster_metadata(self) -> Dict[str, Any]:
        """クラスタリングに関するメタデータを取得する"""
        return {
            'size_thresholds': self.size_thresholds,
            'complexity_thresholds': self.complexity_thresholds,
            'auto_adjusted': self.auto_adjust,
            'language': self.current_language,
            'total_files_classified': self.total_files,
            'cluster_distribution': self.get_cluster_distribution(),
            'cluster_counts': dict(self.cluster_stats)
        }
    
    def get_cluster_coordinates(self, cluster_name: str) -> Tuple[float, float]:
        """クラスタの中央座標（サイズスコア、複雑度スコア）を計算する"""
        # クラスタ名からサイズと複雑度クラスを抽出
        try:
            # 日本語のクラスタ名のフォーマット: "小サイズ・低複雑度"
            parts = cluster_name.split('サイズ・')
            
            size_class = parts[0]
            complexity_class = parts[1].replace('複雑度', '')
            
            # サイズクラスの中央値を計算
            if size_class == self.size_class_labels[self.current_language][0]:  # 小
                size_mid = self.size_thresholds[0] / 2
            elif size_class == self.size_class_labels[self.current_language][1]:  # 中
                size_mid = (self.size_thresholds[0] + self.size_thresholds[1]) / 2
            else:  # 大
                size_mid = self.size_thresholds[1] * 1.5
            
            # 複雑度クラスの中央値を計算
            if complexity_class == self.complexity_class_labels[self.current_language][0]:  # 低
                complexity_mid = self.complexity_thresholds[0] / 2
            elif complexity_class == self.complexity_class_labels[self.current_language][1]:  # 中
                complexity_mid = (self.complexity_thresholds[0] + self.complexity_thresholds[1]) / 2
            else:  # 高
                complexity_mid = self.complexity_thresholds[1] * 1.5
            
            return (size_mid, complexity_mid)
        except Exception as e:
            logging.error(f"クラスタ座標計算中にエラーが発生しました: {e}")
            return (0, 0)

class ClusterAnalyzer:
    """クラスタリングと分類を担当するクラス"""
    
    def __init__(self, size_thresholds: Tuple[float, float] = None, 
                complexity_thresholds: Tuple[float, float] = None,
                auto_adjust: bool = False):
        # デフォルトのしきい値設定
        self.size_thresholds = size_thresholds or (50, 150)  # 小≤50 < 中≤150 < 大
        self.complexity_thresholds = complexity_thresholds or (10, 20)  # 低≤10 < 中≤20 < 高
        self.auto_adjust = auto_adjust
        
        # 言語ローカライズマッピング
        self.size_class_labels = {
            'ja': ['小', '中', '大'],
            'en': ['Small', 'Medium', 'Large']
        }
        
        self.complexity_class_labels = {
            'ja': ['低', '中', '高'],
            'en': ['Low', 'Medium', 'High']
        }
        
        # クラスタの統計情報
        self.cluster_stats = defaultdict(int)
        self.total_files = 0
        
        # 現在の言語設定
        self.current_language = 'ja'
    
    def set_language(self, language_code: str) -> None:
        """言語設定を変更する"""
        if language_code in self.size_class_labels:
            self.current_language = language_code
        else:
            logging.warning(f"サポートされていない言語コード '{language_code}'。デフォルト値を使用します。")
    
    def adjust_thresholds(self, file_metrics: List[Dict[str, float]]) -> None:
        """データに基づいて自動的にしきい値を調整する"""
        if not file_metrics or not self.auto_adjust:
            return
            
        try:
            # サイズとスコアの値を抽出
            size_scores = [m.get('size_score', 0) for m in file_metrics if 'size_score' in m]
            complexity_scores = [m.get('complexity_score', 0) for m in file_metrics if 'complexity_score' in m]
            
            if not size_scores or not complexity_scores:
                return
                
            # パーセンタイルベースのしきい値計算
            size_scores.sort()
            complexity_scores.sort()
            
            # 33パーセンタイルと66パーセンタイルを使用
            size_p33 = size_scores[int(len(size_scores) * 0.33)]
            size_p66 = size_scores[int(len(size_scores) * 0.66)]
            
            complexity_p33 = complexity_scores[int(len(complexity_scores) * 0.33)]
            complexity_p66 = complexity_scores[int(len(complexity_scores) * 0.66)]
            
            # しきい値更新（値が合理的な範囲内の場合）
            if 5 <= size_p33 <= 100 and 50 <= size_p66 <= 300:
                self.size_thresholds = (size_p33, size_p66)
            
            if 3 <= complexity_p33 <= 15 and 10 <= complexity_p66 <= 40:
                self.complexity_thresholds = (complexity_p33, complexity_p66)
                
            logging.info(f"しきい値を自動調整: サイズ={self.size_thresholds}, 複雑度={self.complexity_thresholds}")
        except Exception as e:
            logging.error(f"しきい値調整中にエラーが発生しました: {e}")
    
    def classify_into_cluster(self, size_score: float, complexity_score: float) -> str:
        """サイズと複雑度スコアに基づいてクラスタを分類する"""
        try:
            # サイズ分類
            if size_score <= self.size_thresholds[0]:
                size_idx = 0  # 小
            elif size_score <= self.size_thresholds[1]:
                size_idx = 1  # 中
            else:
                size_idx = 2  # 大
            
            # 複雑度分類
            if complexity_score <= self.complexity_thresholds[0]:
                complexity_idx = 0  # 低
            elif complexity_score <= self.complexity_thresholds[1]:
                complexity_idx = 1  # 中
            else:
                complexity_idx = 2  # 高
            
            size_class = self.size_class_labels[self.current_language][size_idx]
            complexity_class = self.complexity_class_labels[self.current_language][complexity_idx]
            
            # クラスタ名の生成
            cluster_name = f"{size_class}サイズ・{complexity_class}複雑度"
            
            # 統計更新
            self.cluster_stats[cluster_name] += 1
            self.total_files += 1
            
            return cluster_name
        except Exception as e:
            logging.error(f"クラスタ分類中にエラーが発生しました: {e}")
            return "分類不能"
    
    def get_cluster_thresholds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """クラスタ分類のしきい値を取得する"""
        return self.size_thresholds, self.complexity_thresholds
    
    def get_cluster_distribution(self) -> Dict[str, float]:
        """クラスタの分布率を計算する"""
        if self.total_files == 0:
            return {}
        
        return {cluster: (count / self.total_files) * 100 for cluster, count in self.cluster_stats.items()}
    
    def get_cluster_metadata(self) -> Dict[str, Any]:
        """クラスタリングに関するメタデータを取得する"""
        return {
            'size_thresholds': self.size_thresholds,
            'complexity_thresholds': self.complexity_thresholds,
            'auto_adjusted': self.auto_adjust,
            'language': self.current_language,
            'total_files_classified': self.total_files,
            'cluster_distribution': self.get_cluster_distribution(),
            'cluster_counts': dict(self.cluster_stats)
        }
    
    def get_cluster_coordinates(self, cluster_name: str) -> Tuple[float, float]:
        """クラスタの中央座標（サイズスコア、複雑度スコア）を計算する"""
        # クラスタ名からサイズと複雑度クラスを抽出
        try:
            # 日本語のクラスタ名のフォーマット: "小サイズ・低複雑度"
            parts = cluster_name.split('サイズ・')
            
            size_class = parts[0]
            complexity_class = parts[1].replace('複雑度', '')
            
            # サイズクラスの中央値を計算
            if size_class == self.size_class_labels[self.current_language][0]:  # 小
                size_mid = self.size_thresholds[0] / 2
            elif size_class == self.size_class_labels[self.current_language][1]:  # 中
                size_mid = (self.size_thresholds[0] + self.size_thresholds[1]) / 2
            else:  # 大
                size_mid = self.size_thresholds[1] * 1.5
            
            # 複雑度クラスの中央値を計算
            if complexity_class == self.complexity_class_labels[self.current_language][0]:  # 低
                complexity_mid = self.complexity_thresholds[0] / 2
            elif complexity_class == self.complexity_class_labels[self.current_language][1]:  # 中
                complexity_mid = (self.complexity_thresholds[0] + self.complexity_thresholds[1]) / 2
            else:  # 高
                complexity_mid = self.complexity_thresholds[1] * 1.5
            
            return (size_mid, complexity_mid)
        except Exception as e:
            logging.error(f"クラスタ座標計算中にエラーが発生しました: {e}")
            return (0, 0)


class ReportGenerator:
    """レポート生成と可視化を担当するクラス"""
    
    def __init__(self, cluster_analyzer: ClusterAnalyzer):
        self.cluster_analyzer = cluster_analyzer
        # プロット色の設定
        self.plot_colors = {
            'background': '#f5f5f5',
            'grid': '#cccccc',
            'threshold': {
                'size': '#ff6666',
                'complexity': '#6666ff'
            },
            'cluster_box': 'white',
            'text': '#333333'
        }
        # グラフスタイル設定
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 英語ラベルのマッピング
        self.label_map = {
            'サイズスコア': 'Size Score',
            'サイズ境界値': 'Size Threshold',
            '複雑度スコア': 'Complexity Score',
            '複雑度境界値': 'Complexity Threshold',
            '複雑度/サイズ比': 'Complexity/Size Ratio',
            'SQLファイル複雑度・サイズ分析\n3×3クラスタマトリックス': 'SQL File Complexity & Size Analysis\n3×3 Cluster Matrix',
            '分析ファイル数': 'Files Analyzed',
            '平均複雑度': 'Avg Complexity',
            '平均サイズ': 'Avg Size',
            '最も複雑なファイル': 'Most Complex File',
            '最も大きいファイル': 'Largest File',
            '最多クラスタ': 'Most Common Cluster',
            '全SQLファイルにおけるCRUD操作の分布': 'CRUD Operation Distribution in SQL Files',
            '操作数': 'Operation Count'
        }
    
    def generate_metrics_dataframe(self, analysis_results: List[FileAnalysisResult]) -> pd.DataFrame:
        """解析結果をDataFrameに変換する"""
        if not analysis_results:
            return pd.DataFrame()
            
        data = []
        for result in analysis_results:
            # CRUD操作カウント
            crud_counts = {
                'CREATE操作数': sum(1 for ops in result.crud_operations.values() if CRUDOperation.CREATE in ops),
                'READ操作数': sum(1 for ops in result.crud_operations.values() if CRUDOperation.READ in ops),
                'UPDATE操作数': sum(1 for ops in result.crud_operations.values() if CRUDOperation.UPDATE in ops),
                'DELETE操作数': sum(1 for ops in result.crud_operations.values() if CRUDOperation.DELETE in ops)
            }
            
            # 複雑度メトリクス
            metrics = {
                'ファイル名': result.file_name,
                'ファイルパス': result.file_path,
                '複雑度スコア': result.complexity_metrics.complexity_score,
                'サイズスコア': result.complexity_metrics.size_score,
                'SQL文数': result.complexity_metrics.statement_count,
                'サブクエリ数': result.complexity_metrics.subquery_count,
                'JOIN数': result.complexity_metrics.join_count,
                'WHERE条件数': result.complexity_metrics.where_condition_count,
                'CASE文数': result.complexity_metrics.case_count,
                'CTE数': result.complexity_metrics.cte_count,
                '関数呼び出し数': result.complexity_metrics.function_count,
                'テーブル数': result.complexity_metrics.unique_table_count,
                '行数': result.complexity_metrics.total_lines,
                '実効行数': result.complexity_metrics.non_empty_lines,
                '文字数': result.complexity_metrics.total_chars,
                'クラスタ分類': result.cluster_classification
            }
            
            # 複雑度/サイズ比の計算（ゼロ除算防止）
            size_score = max(0.1, metrics['サイズスコア'])  # 0除算防止
            metrics['複雑度/サイズ比'] = metrics['複雑度スコア'] / size_score
            
            # すべての情報をマージ
            # CRUD対象テーブルの一覧を生成 (例: CREATE:[t1,t2]; READ:[t3])
            crud_targets = []
            for tbl, ops in result.crud_operations.items():
                ops_sorted = sorted(op.name for op in ops)
                crud_targets.append(f"{','.join(ops_sorted)}:{tbl}")

            row_data = {**metrics, **crud_counts, 'CRUD対象テーブル': ';'.join(crud_targets)}
            data.append(row_data)
        
        # DataFrameに変換
        df = pd.DataFrame(data)

        # --- CSV 出力向けの整形 ---
        # クラスタ分類に先頭番号を付与する（表示順: 大->中->小 の行順を想定し、左から低->中->高）
        size_classes = self.cluster_analyzer.size_class_labels[self.cluster_analyzer.current_language]
        complexity_classes = self.cluster_analyzer.complexity_class_labels[self.cluster_analyzer.current_language]
        # display order: 大, 中, 小  so that 小 is last row in matrix
        display_size_classes = list(reversed(size_classes))

        cluster_number_map = {}
        num = 1
        for size in display_size_classes:
            for comp in complexity_classes:
                cluster_name = f"{size}サイズ・{comp}複雑度"
                cluster_number_map[cluster_name] = f"{num:02d} {cluster_name}"
                num += 1

        # 既存のクラスタ分類カラムに番号を付与（該当無しはそのまま）
        if 'クラスタ分類' in df.columns:
            df['クラスタ分類'] = df['クラスタ分類'].apply(lambda x: cluster_number_map.get(x, x))

        # 複雑度/サイズ比を小数点第3位までで表示（CSV用）
        if '複雑度/サイズ比' in df.columns:
            df['複雑度/サイズ比'] = df['複雑度/サイズ比'].apply(lambda v: f"{float(v):.3f}")

        return df
    
    # ...existing code...
    
    def _translate(self, text):
        """日本語テキストを英語に変換する"""
        return self.label_map.get(text, text)
    
    def generate_cluster_plot(self, df: pd.DataFrame, output_dir: str) -> str:
        """3x3クラスタのプロット図を生成する"""
        try:
            # データフレームの検証
            if df.empty:
                logging.warning("空のデータフレームのためプロットを生成できません")
                return ""
                
            logging.info(f"プロット生成: {len(df)}行のデータがあります")
            logging.debug(f"サイズスコア範囲: {df['サイズスコア'].min()} - {df['サイズスコア'].max()}")
            logging.debug(f"複雑度スコア範囲: {df['複雑度スコア'].min()} - {df['複雑度スコア'].max()}")
            
            size_thresholds, complexity_thresholds = self.cluster_analyzer.get_cluster_thresholds()
            
            # プロットエリアの設定
            plt.figure(figsize=(14, 12))
            plt.set_cmap('viridis')
            
            # 散布図の作成 - 点のサイズはSQL文数に比例
            min_stmt_count = max(1, df['SQL文数'].min())  # 0除算防止
            max_stmt_count = max(min_stmt_count, df['SQL文数'].max())
            
            # 0除算防止と正規化のロバスト化
            if max_stmt_count > min_stmt_count:
                normalized_sizes = 50 + (df['SQL文数'] - min_stmt_count) / (max_stmt_count - min_stmt_count) * 200
            else:
                normalized_sizes = np.ones(len(df)) * 50  # 全て同じサイズ

            # x/y 軸データ: 0 を扱えないため最小値をクリップ
            df_x = df['サイズスコア'].clip(lower=0.1)
            df_y = df['複雑度スコア'].clip(lower=0.1)

            # 複雑度/サイズ比に基づいて色付け
            try:
                # 散布図を描画（xはクリップ済み df_x、色は数値に変換）
                c_vals = pd.to_numeric(df['複雑度/サイズ比'], errors='coerce').fillna(0)
                scatter = plt.scatter(df_x, df_y, 
                                    alpha=0.8, s=normalized_sizes, 
                                    c=c_vals, cmap='plasma')
                logging.info(f"散布図を描画しました: {len(df)}ポイント")
            except Exception as e:
                logging.error(f"散布図描画中にエラーが発生: {e}")
                # フォールバック: 基本的な散布図を描画
                scatter = plt.scatter(df_x, df_y, 
                                    alpha=0.8, s=50, color='blue')
                logging.info("基本的な散布図を代わりに描画しました")
            
            # しきい値線の描画
            plt.axvline(x=size_thresholds[0], color=self.plot_colors['threshold']['size'], 
                       linestyle='--', alpha=0.7, 
                       label=f'{self._translate("サイズ境界値")} 1 ({size_thresholds[0]:.1f})')
            plt.axvline(x=size_thresholds[1], color=self.plot_colors['threshold']['size'], 
                       linestyle='--', alpha=0.7, 
                       label=f'{self._translate("サイズ境界値")} 2 ({size_thresholds[1]:.1f})')
            plt.axhline(y=complexity_thresholds[0], color=self.plot_colors['threshold']['complexity'], 
                       linestyle='--', alpha=0.7, 
                       label=f'{self._translate("複雑度境界値")} 1 ({complexity_thresholds[0]:.1f})')
            plt.axhline(y=complexity_thresholds[1], color=self.plot_colors['threshold']['complexity'], 
                       linestyle='--', alpha=0.7, 
                       label=f'{self._translate("複雑度境界値")} 2 ({complexity_thresholds[1]:.1f})')
            
            # グラフ範囲設定 - マージンを追加
            x_max = max(df['サイズスコア'].max() * 1.1, 200)
            y_max = max(df['複雑度スコア'].max() * 1.1, 30)

            # サイズ軸（x軸）を symlog にすることで、小サイズ領域を視覚的に広げる
            # symlog を使うと 0 に近い値が扱いにくいため、最小値を0.1にクリップ
            df_x = df['サイズスコア'].clip(lower=0.1)
            df_y = df['複雑度スコア'].clip(lower=0.1)
            plt.xscale('symlog', linthresh=1.0)
            plt.yscale('symlog', linthresh=1.0)

            plt.xlim(0.1, x_max)
            plt.ylim(0.1, y_max)
            
            # クラスタラベルの配置 - 英語ラベルを使用
            self._add_cluster_labels(size_thresholds, complexity_thresholds, x_max, y_max)
            
            # カラーバーの追加
            try:
                cbar = plt.colorbar(scatter)
                cbar.set_label(self._translate('複雑度/サイズ比'), fontsize=10)
            except Exception as e:
                logging.error(f"カラーバー追加中にエラー: {e}")
            
            # グラフの体裁設定 - 英語ラベルを使用
            plt.xlabel(self._translate('サイズスコア'), fontsize=12)
            plt.ylabel(self._translate('複雑度スコア'), fontsize=12)
            plt.title(self._translate('SQLファイル複雑度・サイズ分析\n3×3クラスタマトリックス'), fontsize=16)
            plt.grid(True, alpha=0.3, color=self.plot_colors['grid'])
            plt.legend(loc='upper left', fontsize=10)
            
            # サマリー情報を追加
            self._add_summary_box(df, plt.gca())
            
            # プロット保存
            plot_path = os.path.join(output_dir, 'sql_complexity_cluster_plot.png')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
        except Exception as e:
            logging.error(f"クラスタプロット生成中にエラーが発生しました: {e}")
            return ""
    
    def _add_cluster_labels(self, size_thresholds: Tuple[float, float], 
                           complexity_thresholds: Tuple[float, float], 
                           x_max: float, y_max: float):
        """クラスタラベルを配置する (英語ラベルを使用)"""
        try:
            x_positions = [size_thresholds[0]/2, 
                          (size_thresholds[0]+size_thresholds[1])/2, 
                          (size_thresholds[1]+x_max)/2]
            y_positions = [complexity_thresholds[0]/2, 
                          (complexity_thresholds[0]+complexity_thresholds[1])/2, 
                          (complexity_thresholds[1]+y_max)/2]
            
            # 英語のラベルを使用
            size_labels = self.cluster_analyzer.size_class_labels['en']
            complexity_labels = self.cluster_analyzer.complexity_class_labels['en']
            
            # 各クラスタにラベルを付ける
            for i, y_pos in enumerate(y_positions):
                for j, x_pos in enumerate(x_positions):
                    label = f"{size_labels[j]}-{complexity_labels[i]}"
                    plt.text(x_pos, y_pos, label, 
                            ha='center', va='center', fontsize=10, 
                            bbox=dict(boxstyle='round,pad=0.4', 
                                     facecolor=self.plot_colors['cluster_box'], 
                                     alpha=0.8,
                                     edgecolor='gray'))
        except Exception as e:
            logging.error(f"クラスタラベル配置中にエラーが発生しました: {e}")
    
    def _add_summary_box(self, df: pd.DataFrame, ax):
        """統計情報のテキストボックスをプロットに追加する (英語テキストを使用)"""
        try:
            # 主要統計情報の計算
            total_files = len(df)
            avg_complexity = df['複雑度スコア'].mean()
            avg_size = df['サイズスコア'].mean()
            max_complexity_file = df.loc[df['複雑度スコア'].idxmax()]['ファイル名']
            max_size_file = df.loc[df['サイズスコア'].idxmax()]['ファイル名']
            
            # クラスタ分布の計算
            cluster_dist = df['クラスタ分類'].value_counts()
            cluster_dist_pct = cluster_dist / total_files * 100
            most_common_cluster = cluster_dist.index[0] if not cluster_dist.empty else "None"
                        
        except Exception as e:
            logging.error(f"サマリー情報追加中にエラーが発生しました: {e}")
    
    def generate_crud_distribution_chart(self, df: pd.DataFrame, output_dir: str) -> str:
        """CRUD操作の分布を示す棒グラフを生成する (英語ラベルを使用)"""
        try:
            plt.figure(figsize=(10, 6))
            
            # 各ファイルのCRUD操作数を集計
            crud_data = {
                'CREATE': df['CREATE操作数'].sum(),
                'READ': df['READ操作数'].sum(),
                'UPDATE': df['UPDATE操作数'].sum(),
                'DELETE': df['DELETE操作数'].sum()
            }
            
            # 棒グラフの生成
            colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
            plt.bar(crud_data.keys(), crud_data.values(), color=colors)
            
            # グラフの装飾 - 英語ラベルを使用
            plt.title(self._translate('全SQLファイルにおけるCRUD操作の分布'), fontsize=14)
            plt.ylabel(self._translate('操作数'), fontsize=12)
            plt.grid(axis='y', alpha=0.3)
            
            # 各バーに値を表示
            for i, (op, count) in enumerate(crud_data.items()):
                plt.text(i, count + 5, str(count), ha='center', fontsize=10)
            
            # 保存
            chart_path = os.path.join(output_dir, 'sql_crud_distribution.png')
            plt.tight_layout()
            plt.savefig(chart_path, dpi=300)
            plt.close()
            
            return chart_path
        except Exception as e:
            logging.error(f"CRUD分布図生成中にエラーが発生しました: {e}")
            return ""
    
    
    def generate_markdown_report(self, df: pd.DataFrame, cluster_plot_path: str, 
                                crud_chart_path: str, output_dir: str) -> str:
        """Markdownフォーマットのレポートを生成する"""
        try:
            # クラスタごとのファイル数を集計
            # 以前の処理でクラスタ名に先頭番号を付与しているため、カウント時は先頭番号を剥がす
            cluster_series = df['クラスタ分類'].astype(str).apply(lambda x: re.sub(r'^\d{2}\s+', '', x))
            cluster_counts = cluster_series.value_counts()
            
            # 複雑度とサイズに基づいてトップ10ファイルを抽出
            top_complex = df.nlargest(10, '複雑度スコア')[['ファイル名', '複雑度スコア', 'サイズスコア', 'クラスタ分類']]
            top_size = df.nlargest(10, 'サイズスコア')[['ファイル名', '複雑度スコア', 'サイズスコア', 'クラスタ分類']]
            
            # 基本統計量を計算
            # 要求: サブクエリ数、サイズスコアは不要。代わりに CRUD 各操作数 と 行数 を表示する
            stats_columns = ['複雑度スコア', 'SQL文数', 'JOIN数',
                             'CREATE操作数', 'READ操作数', 'UPDATE操作数', 'DELETE操作数', '行数']
            # 存在しない列がある場合は0列で埋める
            for col in stats_columns:
                if col not in df.columns:
                    df[col] = 0
            stats = df[stats_columns].describe()
            
            # Markdownコンテンツ生成
            md_content = f"""# SQLファイル解析レポート
    
## 解析サマリー
    
- **解析ファイル数**: {len(df)}
- **平均複雑度スコア**: {df['複雑度スコア'].mean():.2f}
- **平均サイズスコア**: {df['サイズスコア'].mean():.2f}
- **最も一般的なクラスタ**: {cluster_counts.index[0] if len(cluster_counts) > 0 else "なし"} ({cluster_counts.iloc[0] if len(cluster_counts) > 0 else 0}ファイル)
    
## クラスタ分析
    
![クラスタ分析図]({os.path.basename(cluster_plot_path)})
    
### クラスタ分布（マトリクス形式）
    
"""
            # 3×3マトリクス形式でクラスタ分布を表示
            # サイズクラスと複雑度クラスを取得
            size_classes = self.cluster_analyzer.size_class_labels[self.cluster_analyzer.current_language]
            complexity_classes = self.cluster_analyzer.complexity_class_labels[self.cluster_analyzer.current_language]
            # 表示順序: 大 -> 中 -> 小 として、小サイズを下の行に並べる
            display_size_classes = list(reversed(size_classes))
            
            # マトリクステーブルのヘッダー
            md_content += f"| サイズ/複雑度 | {complexity_classes[0]}複雑度 | {complexity_classes[1]}複雑度 | {complexity_classes[2]}複雑度 | 合計 |\n"
            md_content += "|------------|------------|------------|------------|------|\n"
            
            # 各行（サイズ）のデータを計算
            total_files = len(df)
            
            for size_idx, size_class in enumerate(display_size_classes):
                row_counts = []
                row_sum = 0
                
                # 各列（複雑度）のデータを計算
                for complexity_idx, complexity_class in enumerate(complexity_classes):
                    cluster_name = f"{size_class}サイズ・{complexity_class}複雑度"
                    count = cluster_counts.get(cluster_name, 0)
                    percentage = (count / total_files * 100) if total_files > 0 else 0
                    row_counts.append(f"{count} ({percentage:.1f}%)")
                    row_sum += count
                
                # 行合計を計算
                row_percentage = (row_sum / total_files * 100) if total_files > 0 else 0
                
                # 行を追加
                md_content += f"| **{size_class}サイズ** | {row_counts[0]} | {row_counts[1]} | {row_counts[2]} | **{row_sum} ({row_percentage:.1f}%)** |\n"
            
            # 列合計を計算
            md_content += "| **合計** |"
            for complexity_idx, complexity_class in enumerate(complexity_classes):
                # 列合計は元のサイズクラス一覧を使って計算
                col_sum = sum(cluster_counts.get(f"{size_class}サイズ・{complexity_class}複雑度", 0) for size_class in size_classes)
                col_percentage = (col_sum / total_files * 100) if total_files > 0 else 0
                md_content += f" **{col_sum} ({col_percentage:.1f}%)** |"
            md_content += f" **{total_files} (100.0%)** |\n\n"
            
            # --- 統計情報セクションは削除しました。代わりに CRUD サマリを出力します ---
            # 全体の CRUD 件数
            total_create = int(df['CREATE操作数'].sum()) if 'CREATE操作数' in df.columns else 0
            total_read = int(df['READ操作数'].sum()) if 'READ操作数' in df.columns else 0
            total_update = int(df['UPDATE操作数'].sum()) if 'UPDATE操作数' in df.columns else 0
            total_delete = int(df['DELETE操作数'].sum()) if 'DELETE操作数' in df.columns else 0

            md_content += "\n## CRUD サマリ\n\n"
            md_content += f"- **全体 CREATE 件数**: {total_create}\n"
            md_content += f"- **全体 READ 件数**: {total_read}\n"
            md_content += f"- **全体 UPDATE 件数**: {total_update}\n"
            md_content += f"- **全体 DELETE 件数**: {total_delete}\n\n"

            # テーブルごとの CRUD 件数を集計
            # 可能なら metrics の CRUD カウントカラムを使い、なければ 'CRUD対象テーブル' を解析
            table_crud = defaultdict(lambda: {'CREATE': 0, 'READ': 0, 'UPDATE': 0, 'DELETE': 0})
            if {'CREATE操作数', 'READ操作数', 'UPDATE操作数', 'DELETE操作数'}.issubset(df.columns):
                # metrics にファイル毎の CRUD 合計があるので、テーブル単位集計は CRUD対象テーブル を使う
                pass

            # CRUD対象テーブル列がある場合はパースしてテーブルごとにカウント
            if 'CRUD対象テーブル' in df.columns:
                for _, row in df.iterrows():
                    crud_field = row.get('CRUD対象テーブル', '')
                    if not crud_field:
                        continue
                    parts = [p for p in str(crud_field).split(';') if p]
                    for part in parts:
                        if ':' in part:
                            ops_part, tbl = part.split(':', 1)
                            ops = [op.strip().upper() for op in ops_part.split(',') if op.strip()]
                            # 正規化したテーブル名を得る
                            tbl_clean = self._normalize_table_name_for_summary(tbl)
                            for op in ops:
                                if op == 'CREATE':
                                    table_crud[tbl_clean]['CREATE'] += 1
                                elif op == 'READ':
                                    table_crud[tbl_clean]['READ'] += 1
                                elif op == 'UPDATE':
                                    table_crud[tbl_clean]['UPDATE'] += 1
                                elif op == 'DELETE':
                                    table_crud[tbl_clean]['DELETE'] += 1

            # テーブルごとの集計を Markdown テーブルで出力（上位 N を表示）
            if table_crud:
                md_content += "### テーブル別 CRUD 件数（上位100）\n\n"
                md_content += "| テーブル名 | CREATE | READ | UPDATE | DELETE | 合計 |\n"
                md_content += "|---|---:|---:|---:|---:|---:|\n"
                # 合計でソート
                sorted_tables = sorted(table_crud.items(), key=lambda kv: sum(kv[1].values()), reverse=True)
                for tbl, counts in sorted_tables[:100]:
                    tot = counts['CREATE'] + counts['READ'] + counts['UPDATE'] + counts['DELETE']
                    md_content += f"| {tbl} | {counts['CREATE']} | {counts['READ']} | {counts['UPDATE']} | {counts['DELETE']} | {tot} |\n"
                md_content += "\n"
            md_content += f"""
---
*このレポートは自動生成されました。生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            
            # Markdownファイルとして保存
            md_path = os.path.join(output_dir, 'sql_analysis_report.md')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            return md_path
        except Exception as e:
            logging.error(f"Markdownレポート生成中にエラーが発生しました: {e}")
            return ""

    def _normalize_table_name_for_summary(self, raw: str) -> str:
        """CRUDサマリ用のテーブル名正規化: 引用符除去・小文字化・スキーマ部分は含めない"""
        try:
            if not raw:
                return ''
            # remove quotes and brackets
            t = re.sub(r'["\[\]`\']', '', raw)
            t = t.strip()
            # if schema.table, take table part
            if '.' in t:
                parts = [p for p in t.split('.') if p]
                t = parts[-1]
            return t.lower()
        except Exception:
            return raw.strip()
    
    def generate_comprehensive_report(self, df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
        """すべてのレポートを生成し、ファイルパスを返す"""
        try:
            # 各種プロットを生成
            cluster_plot_path = self.generate_cluster_plot(df, output_dir)
            crud_chart_path = self.generate_crud_distribution_chart(df, output_dir)
            
            # Markdownレポートを生成
            md_report_path = self.generate_markdown_report(df, cluster_plot_path, crud_chart_path, output_dir)
            
            return {
                'metrics_csv': os.path.join(output_dir, 'sql_comprehensive_metrics.csv'),
                'cluster_plot': cluster_plot_path,
                'crud_chart': crud_chart_path,
                'markdown_report': md_report_path
            }
        except Exception as e:
            logging.error(f"総合レポート生成中にエラーが発生しました: {e}")
            return {}
    

class SQLAnalysisManager:
    """SQL解析の全体統制とワークフロー管理を担当するクラス"""
    
    def __init__(self, directory_path: str):
        self.directory_path = directory_path
        self.file_reader = SQLFileReader()
        self.parser = SQLParser()
        self.crud_analyzer = CRUDAnalyzer(self.parser)
        self.complexity_analyzer = ComplexityAnalyzer(self.parser)
        self.cluster_analyzer = ClusterAnalyzer()
        self.report_generator = ReportGenerator(self.cluster_analyzer)
        self.analysis_results = []
    
    def execute_comprehensive_analysis(self) -> List[FileAnalysisResult]:
        """包括的解析を実行する"""
        sql_files = self.file_reader.find_sql_files(self.directory_path)

        def analyze_file(sql_file: str) -> Optional[FileAnalysisResult]:
            try:
                content = self.file_reader.read_file_content(sql_file)
                if content is None:
                    return None

                crud_operations = self.crud_analyzer.extract_crud_operations(content)
                complexity_metrics = self.complexity_analyzer.calculate_complexity_metrics(content)
                cluster_classification = self.cluster_analyzer.classify_into_cluster(
                    complexity_metrics.size_score,
                    complexity_metrics.complexity_score
                )

                return FileAnalysisResult(
                    file_name=os.path.basename(sql_file),
                    file_path=sql_file,
                    content=content,
                    crud_operations=crud_operations,
                    complexity_metrics=complexity_metrics,
                    cluster_classification=cluster_classification
                )
            except Exception as e:
                logging.debug(f"ファイル解析中に例外: {sql_file}: {e}")
                return None

        # max_workers can be overridden by env var for benchmarking (e.g. SQL_ANALYZER_MAX_WORKERS=1)
        try:
            env_workers = os.getenv('SQL_ANALYZER_MAX_WORKERS')
            if env_workers:
                max_workers = max(1, int(env_workers))
            else:
                max_workers = min(8, max(2, (os.cpu_count() or 2)))
        except Exception:
            max_workers = min(8, max(2, (os.cpu_count() or 2)))
        results: List[Optional[FileAnalysisResult]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(analyze_file, f): f for f in sql_files}
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                if res:
                    results.append(res)

        self.analysis_results.extend(results)
        return self.analysis_results
    
        
    def generate_comprehensive_report(self, output_dir: str = '.') -> None:
        """包括的レポートを生成する"""
        if not self.analysis_results:
            self.execute_comprehensive_analysis()
        
        # DataFrameを生成
        metrics_df = self.report_generator.generate_metrics_dataframe(self.analysis_results)
        
        # CSV保存
        metrics_csv_path = os.path.join(output_dir, 'sql_comprehensive_metrics.csv')
        metrics_df.to_csv(metrics_csv_path, index=False, encoding='utf-8')
        
        # ファイルごとの CRUD 対象テーブルを正規化して別CSVに保存
        try:
            table_targets_rows = []
            for _, row in metrics_df.iterrows():
                file_name = row.get('ファイル名', '')
                crud_field = row.get('CRUD対象テーブル', '')
                if not crud_field:
                    continue
                # 例: "CREATE,READ:customers;UPDATE:orders"
                parts = [p for p in crud_field.split(';') if p]
                for part in parts:
                    if ':' in part:
                        ops_part, tbl = part.split(':', 1)
                        ops = [op.strip() for op in ops_part.split(',') if op.strip()]
                        for op in ops:
                            # テーブル名を正規化し、スキーマとテーブルを分離
                            cleaned = self.crud_analyzer._clean_table_name(tbl) if hasattr(self, 'crud_analyzer') else tbl.strip()
                            if not cleaned:
                                continue
                            if '.' in cleaned:
                                schema, table = cleaned.split('.', 1)
                            else:
                                schema, table = '', cleaned
                            table_targets_rows.append({'ファイル名': file_name, '操作': op, 'スキーマ': schema, 'テーブル名': table})
                    else:
                        # 予期しないフォーマットは可能な限り正規化して出力
                        cleaned = self.crud_analyzer._clean_table_name(part) if hasattr(self, 'crud_analyzer') else part.strip()
                        if cleaned:
                            if '.' in cleaned:
                                schema, table = cleaned.split('.', 1)
                            else:
                                schema, table = '', cleaned
                            table_targets_rows.append({'ファイル名': file_name, '操作': '', 'スキーマ': schema, 'テーブル名': table})
                        else:
                            table_targets_rows.append({'ファイル名': file_name, '操作': '', 'スキーマ': '', 'テーブル名': part})

            if table_targets_rows:
                table_targets_df = pd.DataFrame(table_targets_rows)
                # カラム順を整える
                cols = ['ファイル名', 'スキーマ', 'テーブル名', '操作']
                for c in cols:
                    if c not in table_targets_df.columns:
                        table_targets_df[c] = ''
                table_targets_df = table_targets_df[cols]
                table_targets_csv_path = os.path.join(output_dir, 'sql_table_targets.csv')
                table_targets_df.to_csv(table_targets_csv_path, index=False, encoding='utf-8')
                logging.info(f"CRUD対象テーブル一覧を {table_targets_csv_path} に保存しました")
        except Exception as e:
            logging.error(f"CRUD対象テーブルCSV生成中にエラーが発生しました: {e}")
        
        # すべてのレポートを生成
        report_paths = self.report_generator.generate_comprehensive_report(metrics_df, output_dir)
        
        logging.info(f"包括的解析レポートが {output_dir} に保存されました:")
        for report_type, path in report_paths.items():
            if path:
                logging.info(f"- {path}: {report_type}")
        
        if self.file_reader.failed_files:
            logging.warning(f"注意: {len(self.file_reader.failed_files)}ファイルの解析に失敗")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='モジュラー設計SQL解析ツール')
    parser.add_argument('directory', help='解析対象SQLファイルディレクトリパス')
    parser.add_argument('-o', '--output', default='./output', help='レポート出力先ディレクトリ (デフォルト: ./output)')
    parser.add_argument('-v', '--verbose', action='store_true', help='詳細ログ表示')
    
    args = parser.parse_args()

    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    if not os.path.isdir(args.directory):
        logging.error(f"ディレクトリ {args.directory} が存在しない")
        sys.exit(1)
    
    # 出力ディレクトリがなければ作成
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # 解析実行
    manager = SQLAnalysisManager(args.directory)
    
    if args.verbose:
        logging.debug("包括的SQL解析を開始...")
    
    # レポートは常に生成（デフォルト動作）
    manager.generate_comprehensive_report(args.output)
    
    if args.verbose:
        logging.debug("解析完了")
    else:
        # verboseモードでない場合も、レポート生成を明示的に表示
        logging.info(f"SQL解析レポートが {args.output} に生成されました")

if __name__ == "__main__":
    main()
