import cv2
import cvxpy as cp
import numpy as np
import time

class QueensGame():
    def __init__(self, screenshot_path: str):
        print("Extracting board from screenshot...")
        start_time = time.time()
        board_bw = self._find_board_bounding_box(screenshot_path)
        self._compute_game_size(board_bw)
        self._compute_border_length(board_bw)
        self._compute_square_length(board_bw)
        self._compute_boundary_width(board_bw)
        self._parse_squares()
        self._load_crown()
        print("Extracted in {:.4f} seconds".format(time.time() - start_time))

    def _find_board_bounding_box(self, screenshot_path: str) -> np.ndarray:
        img_color = cv2.imread(screenshot_path)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        img_bw = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)[1]
        largest_square = self._find_largest_square(img_gray)
        x, y, w, h = cv2.boundingRect(largest_square)
        self.board_color = img_color[(y + 1):(y + h - 1), (x + 1):(x + w - 1)]
        board_bw = img_bw[(y + 1):(y + h - 1), (x + 1):(x + w - 1)]
        board_bw = (board_bw.astype(float) / float(255)).astype(int)
        return board_bw
    
    def _find_largest_square(self, img_gray: np.ndarray) -> np.ndarray:
        _, thresh = cv2.threshold(img_gray, 200, 255, 0)
        contours, _ = cv2.findContours(thresh, 1, 2)
        max_width, largest_square = 0, 0
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                _, _, w, h = cv2.boundingRect(cnt)
                ratio = float(w)/h
                if (ratio >= 0.9 and ratio <= 1.1):
                    if w > max_width:
                        max_width = w
                        largest_square = cnt
        return largest_square

    def _compute_game_size(self, board_bw: np.ndarray):
        # Compute the size of the game (an n x n game has size `n`)
        self.n = np.unique(np.diff(np.round(np.mean(board_bw, axis=1))), return_counts=True)[1][0]

    def _compute_border_length(self, board_bw: np.ndarray):
        # Compute length of the board's border
        self.board_border_length = np.where(np.round(np.mean(board_bw, axis=0)) == 1)[0][0]

    def _compute_square_length(self, board_bw: np.ndarray):
        # Compute the length of the side of one of the board's squares
        max_length = 0
        lengths = []
        for i in range(self.board_border_length, len(board_bw) - self.board_border_length):
            row = board_bw[i]
            idx_pairs = np.where(np.diff(np.hstack(([False], row==1,[False]))))[0].reshape(-1,2)
            if len(idx_pairs) != 0:
                curr_max = np.max(np.diff(idx_pairs, axis=1))
                lengths.append(curr_max)
                if curr_max > max_length:
                    max_length = curr_max
        for i in range(self.board_border_length, len(board_bw) - self.board_border_length):
            col = board_bw[:, i]
            idx_pairs = np.where(np.diff(np.hstack(([False], col==1,[False]))))[0].reshape(-1,2)
            if len(idx_pairs) != 0:
                curr_max = np.max(np.diff(idx_pairs, axis=1))
                lengths.append(curr_max)
                if curr_max > max_length:
                    max_length = curr_max
        self.square_len = max_length

    def _compute_boundary_width(self, board_bw: np.ndarray):
        # Compute the width of the non-bolded boundary between each square
        min_length = len(board_bw)
        for i in range(self.board_border_length, len(board_bw) - self.board_border_length):
            row = board_bw[i]
            idx_pairs = np.where(np.diff(np.hstack(([False], row==0, [False]))))[0].reshape(-1,2)
            if len(idx_pairs) != 0:
                curr_min = np.min(np.diff(idx_pairs, axis=1))
                if curr_min < min_length:
                    min_length = curr_min
        for i in range(self.board_border_length, len(board_bw) - self.board_border_length):
            col = board_bw[:, i]
            idx_pairs = np.where(np.diff(np.hstack(([False], col==0, [False]))))[0].reshape(-1,2)
            if len(idx_pairs) != 0:
                curr_min = np.min(np.diff(idx_pairs, axis=1))
                if curr_min < min_length:
                    min_length = curr_min
        self.boundary_width = min_length

    def _parse_squares(self):
        # Find each square, its color, and create the regions
        self.squares = np.zeros(shape=(self.n, self.n, 2), dtype=int)
        self.regions = {}
        start_idx = np.array([self.board_border_length, self.board_border_length])
        for i in range(self.n):
            for j in range(self.n):
                upper_left = start_idx + np.array([i, j]) * (self.square_len + self.boundary_width)
                center = upper_left + (self.square_len // 2).astype(int)
                color = tuple(self.board_color[*center])
                if color not in self.regions:
                    self.regions[color] = set()
                self.regions[color].add((i, j))
                self.squares[i, j] = upper_left

    def _load_crown(self):
        self.crown_mask = cv2.imread("./images/crown.png")
        self.crown_mask = cv2.resize(self.crown_mask, (self.square_len - self.boundary_width, self.square_len - self.boundary_width))
        self.crown_mask = (self.crown_mask / float(255)).astype(int)

    def solve(self, save_to_path: str):
        print("Solving Queens...")
        crowns = cp.Variable(shape=(self.n, self.n), boolean=True)

        # 1 Crown per column
        constraints = [cp.sum([crowns[i, j] for i in range(self.n)]) == 1 for j in range(self.n)]
        # 1 Crown per row
        constraints += [cp.sum([crowns[i, j] for j in range(self.n)]) == 1 for i in range(self.n)]
        # 1 Crown per region
        for region in self.regions.values():
            constraints += [cp.sum([crowns[i, j] for (i, j) in region]) == 1]
        ## Two crowns cannot touch each other, not even diagonally
        last = self.n - 1
        for i in range(self.n):
            down = i + 1
            for j in range(self.n):
                left, right = j - 1, j + 1
                if (i < last):
                    constraints += [crowns[i, j] + crowns[down, j] <= 1]
                    if (j > 0):
                        constraints += [crowns[i, j] + crowns[down, left] <= 1]
                    if (j < last):
                        constraints += [crowns[i, j] + crowns[down, right] <= 1]
                if (j < last):
                    constraints += [crowns[i, j] + crowns[i, right] <= 1]

        problem = cp.Problem(cp.Minimize(0), constraints)
        start_time = time.time()
        problem.solve()
        print("Solved in {:.4f} seconds".format(time.time() - start_time))
        self._save_solution(save_to_path, crowns.value)
        print("Solution saved to " + save_to_path)

    def _save_solution(self, save_to_path: str, crown_locations: np.ndarray):
        solution = self.board_color.copy()
        for i in range(self.n):
            for j in range(self.n):
                if crown_locations[i, j]:
                    start_idx = self.squares[i, j]
                    end_idx = self.squares[i, j] + self.square_len
                    solution[(start_idx[0]):(end_idx[0] - self.boundary_width), (start_idx[1]):(end_idx[1] - self.boundary_width)] = solution[(start_idx[0]):(end_idx[0] - self.boundary_width), (start_idx[1]):(end_idx[1] - self.boundary_width)] * self.crown_mask
        cv2.imwrite(save_to_path, solution)

#### Alternate Solution

# If there is any row or column with only one color available, then the crown for that color has to be in that row or column
    #   -> Place X's everywhere else in that region except for that row or column
    #   -> Note: This is just an instance of the 3rd rule
# If the available squares for a region are in a single row/col, then the crown for that row/col has to be in that region
    #   -> Place X's in every other square in that row/col
    #   -> Note: This is just an instance of the rule below

# See if squares across multiple regions occupy the same number of rows/columns as regions
    #   -> Place X's in all other squares in those rows/cols

# Run a scenario for each of the different squares in a region (one-step lookahead)
    #   -> if any square is X-ed out for all the scenarios, then place an X in that square
    #   -> if placing a crown in a square results in a column/row/region with all X's, then place an X in that square

    # Can do a shortcut for identified patterns:
    #   -> Straight Line
    #   -> W shape
    #   -> L shape
    #   -> Corner
    #   -> 2 adjacent squares

# If there is one empty space in a column, row, or region, then:
    #   -> Place a crown in that empty space
    #   -> Place X's in each of the adjacent squares, and in the remainder of the row, column, and region

# import numpy as np

# from color import Color
# from games import game_50

# game = game_50
# n = len(game)

# X = np.zeros(shape=(n, n))

# def axis_all_one_color(X: np.ndarray, C: np.ndarray, regions: dict[Color: set[tuple[int, int]]], axis: int):
#     """axis: 0, means looking down columns (across rows); 1, means looking down rows (across columns)"""
#     for i in range(len(X)):
#         if axis == 0:
#             color = C[0, i]
#             all_one_color = np.all(C[:, i] == color)
#         else:  # axis == 1
#             color = C[i, 0]
#             all_one_color = np.all(C[i] == color)
        
#         if all_one_color:
            