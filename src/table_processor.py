import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from PIL import Image
import pytesseract

@dataclass
class TableCell:
    content: str
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    confidence: float = 1.0
    formatting: Dict = None

@dataclass
class Table:
    cells: List[TableCell]
    num_rows: int
    num_cols: int
    title: Optional[str] = None
    metadata: Dict = None

class TableProcessor:
    def __init__(self, tesseract_path: Optional[str] = None):
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        self.table_detection_params = {
            'min_cell_height': 20,
            'min_cell_width': 40,
            'min_rows': 2,
            'min_cols': 2,
            'line_thickness_range': (1, 5)
        }

    def detect_tables(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        horizontal = self._detect_lines(binary, 'horizontal')
        vertical = self._detect_lines(binary, 'vertical')

        table_mask = cv2.bitwise_or(horizontal, vertical)

        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (w > self.table_detection_params['min_cell_width'] * 2 and
                h > self.table_detection_params['min_cell_height'] * 2):
                tables.append((x, y, w, h))

        return tables

    def _detect_lines(self, binary: np.ndarray, direction: str) -> np.ndarray:
        if direction == 'horizontal':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        eroded = cv2.erode(binary, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)

        return dilated

    def detect_merged_cells(self, table_region: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect merged cells in table using contour analysis"""
        gray = cv2.cvtColor(table_region, cv2.COLOR_BGR2GRAY)
        
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        merged_cells = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Проверка на объединенную ячейку по размеру
            if w > self.table_detection_params['min_cell_width'] * 1.5 or \
               h > self.table_detection_params['min_cell_height'] * 1.5:
                merged_cells.append((x, y, w, h))
        
        return merged_cells

    def analyze_table_data(self, table: Table) -> Dict:
        """Analyze data in table and extract insights"""
        # Преобразование в pandas DataFrame для анализа
        data = {}
        headers = []
        for cell in table.cells:
            if cell.row == 0:
                headers.append(cell.content)
                data[cell.content] = [None] * (table.num_rows - 1)
            else:
                if len(headers) > cell.col:
                    header = headers[cell.col]
                    data[header][cell.row - 1] = cell.content

        df = pd.DataFrame(data)

        # Анализ данных
        analysis = {
            'summary': {},
            'patterns': [],
            'anomalies': [],
            'relationships': []
        }

        # Базовая статистика
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis['summary']['numeric'] = df[numeric_cols].describe().to_dict()

        # Поиск пропущенных значений
        missing = df.isnull().sum()
        if missing.any():
            analysis['anomalies'].append({
                'type': 'missing_values',
                'details': missing.to_dict()
            })

        # Поиск дубликатов
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            analysis['anomalies'].append({
                'type': 'duplicates',
                'count': int(duplicates)
            })

        # Корреляции между числовыми столбцами
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            strong_corr = np.where(np.abs(corr) > 0.7)
            for i, j in zip(*strong_corr):
                if i != j:
                    analysis['relationships'].append({
                        'type': 'correlation',
                        'columns': [numeric_cols[i], numeric_cols[j]],
                        'value': float(corr.iloc[i, j])
                    })

        return analysis

    def merge_tables(self, tables: List[Table], strategy: str = 'vertical') -> Table:
        """Merge multiple tables into one"""
        if not tables:
            return None

        if strategy == 'vertical':
            # Вертикальное объединение (друг под другом)
            merged_cells = []
            row_offset = 0
            max_cols = max(table.num_cols for table in tables)

            for table in tables:
                for cell in table.cells:
                    new_cell = TableCell(
                        content=cell.content,
                        row=cell.row + row_offset,
                        col=cell.col,
                        rowspan=cell.rowspan,
                        colspan=cell.colspan,
                        confidence=cell.confidence,
                        formatting=cell.formatting.copy() if cell.formatting else None
                    )
                    merged_cells.append(new_cell)
                row_offset += table.num_rows

            return Table(
                cells=merged_cells,
                num_rows=row_offset,
                num_cols=max_cols
            )

        elif strategy == 'horizontal':
            # Горизонтальное объединение (рядом друг с другом)
            merged_cells = []
            col_offset = 0
            total_rows = max(table.num_rows for table in tables)

            for table in tables:
                for cell in table.cells:
                    new_cell = TableCell(
                        content=cell.content,
                        row=cell.row,
                        col=cell.col + col_offset,
                        rowspan=cell.rowspan,
                        colspan=cell.colspan,
                        confidence=cell.confidence,
                        formatting=cell.formatting.copy() if cell.formatting else None
                    )
                    merged_cells.append(new_cell)
                col_offset += table.num_cols

            return Table(
                cells=merged_cells,
                num_rows=total_rows,
                num_cols=col_offset
            )

        else:
            raise ValueError(f'Unsupported merge strategy: {strategy}')

    def export_to_html(self, table: Table) -> str:
        """Export table to HTML format with preserved formatting"""
        html = ['<table border="1" cellpadding="3" cellspacing="0">']        
        current_row = 0
        
        while current_row < table.num_rows:
            html.append('<tr>')
            current_col = 0
            
            while current_col < table.num_cols:
                cell = self._find_cell(table.cells, current_row, current_col)
                
                if cell:
                    style = []
                    if cell.formatting:
                        if 'background_color' in cell.formatting:
                            style.append(
                                f'background-color: {cell.formatting["background_color"]};'
                            )
                        if 'text_align' in cell.formatting:
                            style.append(
                                f'text-align: {cell.formatting["text_align"]};'
                            )
                    
                    attrs = []
                    if cell.rowspan > 1:
                        attrs.append(f'rowspan="{cell.rowspan}"')
                    if cell.colspan > 1:
                        attrs.append(f'colspan="{cell.colspan}"')
                    if style:
                        attrs.append(f'style="{"".join(style)}"')
                    
                    html.append(
                        f'<td {" ".join(attrs)}>{cell.content}</td>'
                    )
                    current_col += cell.colspan
                else:
                    current_col += 1
            
            html.append('</tr>')
            current_row += 1
        
        html.append('</table>')
        return '\n'.join(html)

    def _find_cell(self, cells: List[TableCell], row: int, col: int) -> Optional[TableCell]:
        """Find cell at specified position"""
        for cell in cells:
            if cell.row == row and cell.col == col:
                return cell
        return None

    def extract_table_structure(self, image: np.ndarray, box: Tuple[int, int, int, int]) -> Table:
        """Extract table structure from image region"""
        x, y, w, h = box
        table_region = image[y:y+h, x:x+w]

        # Обнаружение линий
        gray = cv2.cvtColor(table_region, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        horizontal = self._detect_lines(binary, 'horizontal')
        vertical = self._detect_lines(binary, 'vertical')

        # Нахождение пересечений
        intersections = cv2.bitwise_and(horizontal, vertical)
        intersection_points = cv2.findNonZero(intersections)

        if intersection_points is None:
            return None

        # Сортировка точек пересечения
        points = intersection_points.reshape(-1, 2)
        sorted_points = points[np.lexsort((points[:, 0], points[:, 1]))]

        # Определение структуры таблицы
        rows = self._group_points(sorted_points[:, 1])
        num_rows = len(rows)

        cells = []
        for i, row_points in enumerate(rows[:-1]):
            row_cells = self._extract_row_cells(
                table_region,
                row_points,
                rows[i + 1],
                i
            )
            cells.extend(row_cells)

        num_cols = max(cell.col for cell in cells) + 1 if cells else 0

        return Table(
            cells=cells,
            num_rows=num_rows - 1,
            num_cols=num_cols,
            metadata={'location': box}
        )
