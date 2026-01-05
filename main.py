"""
Continuous screen monitoring service for N-Queens puzzle detection
"""
import time
import subprocess
import cv2
import numpy as np
from PIL import Image
import mss

from constants import data_dir


MIN_AREA = 1000
MIN_ASPECT_RATIO = 0.95

EDGE_BLACK_THRESH = 30


def find_grid_contour(gray, min_area=MIN_AREA, min_aspect_ratio=MIN_ASPECT_RATIO):
    """
    Detect and localize a potential grid in the image. Return all plausible candidates.
    """
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find grid candidates
    grid_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:  # Too small
            continue

        # Look for rectangles or near-rectangles
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) < 4:
            continue

        # Grids are usually squarish
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if not (min_aspect_ratio < aspect_ratio < 1 / min_aspect_ratio):
            continue

        if not validate_edge_black(gray, x, y, w, h):
            continue

        grid_candidates.append((x, y, w, h))

    return grid_candidates


def validate_edge_black(gray, x, y, w, h, edge_black_threshold=EDGE_BLACK_THRESH):
    """
    Validate if the edge of the rectangle given by x,y,w,h is mostly black
    """
    # Clamp bounds to valid range
    y_start = max(0, y)
    y_end = min(gray.shape[0], y + h)
    x_start = max(0, x)
    x_end = min(gray.shape[1], x + w)

    # Get all edge pixels
    edge_pixels = np.concatenate([
        gray[y_start, x_start:x_end],              # Top edge
        gray[y_end - 1, x_start:x_end],            # Bottom edge
        gray[y_start + 1:y_end - 1, x_start],      # Left edge (exclude corners)
        gray[y_start + 1:y_end - 1, x_end - 1]     # Right edge (exclude corners)
    ])

    edge_is_black = np.mean(edge_pixels) < edge_black_threshold

    return edge_is_black


def find_grid_cells(original, gray, x, y, w, h, black_threshold=40, black_ratio=0.6):
    """
    Find grid lines by scanning rows/columns for black content.
    Then find regular spacing pattern.
    Input is assumed to already be the image of the entire grid.
    """
    # Extract grid region
    grid_roi = gray[y:y+h, x:x+w]

    # grid_roi_original = original[y:y+h, x:x+w]
    # cv2.imshow('Debug - grid_roi_original', grid_roi_original)
    # cv2.waitKey(0)  # Wait for key press
    # cv2.destroyAllWindows()


    def find_grid_lines(projection_axis):
        """
        Find grid lines along one axis.
        projection_axis: 0 for rows (horizontal lines), 1 for cols (vertical lines)
        """
        size = grid_roi.shape[projection_axis]
        is_grid_line = []

        # Step 1: Check each line - is it a grid line?
        for i in range(size):
            if projection_axis == 0:
                line = grid_roi[i, :]  # Row
            else:
                line = grid_roi[:, i]  # Column

            # Count black pixels
            black_pixels = np.sum(line < black_threshold)
            ratio = black_pixels / len(line)

            is_grid_line.append(bool(ratio >= black_ratio))

        # Step 2: Find consecutive chunks of grid lines
        chunks = []
        in_chunk = False
        chunk_start = 0

        for i in range(size):
            if is_grid_line[i] and not in_chunk:
                # Start of chunk
                in_chunk = True
                chunk_start = i
            elif not is_grid_line[i] and in_chunk:
                # End of chunk
                chunk_end = i - 1
                chunk_middle = (chunk_start + chunk_end) // 2
                chunks.append(chunk_middle)
                in_chunk = False

        # Handle chunk at the end
        if in_chunk:
            chunk_middle = (chunk_start + size - 1) // 2
            chunks.append(chunk_middle)

        # Step 3: Find regular spacing pattern
        if len(chunks) < 2:
            return np.array([])

        spacings = np.diff(chunks)
        median_spacing = np.median(spacings)

        # Filter to keep only regularly spaced lines
        regular_lines = [chunks[0]]
        for i in range(1, len(chunks)):
            spacing = chunks[i] - regular_lines[-1]
            # Allow 20% tolerance
            if abs(spacing - median_spacing) / median_spacing < 0.2:
                regular_lines.append(chunks[i])

        return np.array(regular_lines)

    # Find horizontal and vertical grid lines
    row_lines = find_grid_lines(0)  # Horizontal lines
    col_lines = find_grid_lines(1)  # Vertical lines

    if len(row_lines) < 2 or len(col_lines) < 2:
        return None, None

    # Create cells - cells are BETWEEN grid lines + assign color
    n_rows = len(row_lines) - 1
    n_cols = len(col_lines) - 1

    cells = []
    unique_colors = {}
    for i in range(n_rows):
        cells.append([])
        for j in range(n_cols):
            # Cell is between line i and i+1
            cell_top = row_lines[i]
            cell_bottom = row_lines[i + 1]
            cell_left = col_lines[j]
            cell_right = col_lines[j + 1]

            cell = original[y+cell_top:y+cell_bottom, x+cell_left:x+cell_right]
            color = np.median(cell, axis=(0, 1))
            color_hex = color[0] + color[1]*256 + color[2]*256
            if color_hex not in unique_colors:
                unique_colors[color_hex] = len(unique_colors)

            cells[i].append({
                'color': color,
                'color_idx': unique_colors[color_hex],
                'x': x + cell_left,
                'y': y + cell_top,
                'w': cell_right - cell_left,
                'h': cell_bottom - cell_top
            })

    return (n_rows, n_cols), cells


def n_queens_check_partial_sol(perm: list, grid: np.ndarray) -> bool:
    colors = set()
    cols = set()
    for k in range(0, len(perm)):
        color = grid[k][perm[k]]
        if color in colors:
            return False
        colors.add(color)

        if perm[k] in cols:
            return False
        cols.add(perm[k])
    return True


def n_queens_extend_partial_sol(perm: list, n: int):
    new_perm = []
    for p in perm:
        for i in range(0, n):
            if len(p) > 0 and abs(p[-1] - i) <= 1:  # don't add above and in diag
                continue
            new_perm.append(p + [i])
    return new_perm


def n_queens_find_sol(n: int, grid: np.ndarray) -> int:
    perm = [[]]
    for i in range(n):
        new_perm = list(filter(lambda p: n_queens_check_partial_sol(p, grid), n_queens_extend_partial_sol(perm, n)))
        perm = new_perm
    return perm


def visualize_solution(cells, solution):
    """
    Create a grid visualization with cells colored and queens placed.
    """
    n_rows = len(cells)
    n_cols = len(cells[0])

    # Get cell dimensions (assuming uniform)
    cell_w = cells[0][0]['w']
    cell_h = cells[0][0]['h']

    # Create image
    img_w = n_cols * cell_w
    img_h = n_rows * cell_h
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # Fill cells with colors
    for i in range(n_rows):
        for j in range(n_cols):
            cell = cells[i][j]
            y_start = i * cell_h
            y_end = (i + 1) * cell_h
            x_start = j * cell_w
            x_end = (j + 1) * cell_w

            img[y_start:y_end, x_start:x_end] = cell['color']

    # Draw grid lines (black)
    for i in range(n_rows + 1):
        y = i * cell_h
        cv2.line(img, (0, y), (img_w, y), (0, 0, 0), 2)

    for j in range(n_cols + 1):
        x = j * cell_w
        cv2.line(img, (x, 0), (x, img_h), (0, 0, 0), 2)

    # Draw queens
    for row, col in enumerate(solution):
        center_x = col * cell_w + cell_w // 2
        center_y = row * cell_h + cell_h // 2
        radius = min(cell_w, cell_h) // 5

        cv2.circle(img, (center_x, center_y), radius, (0, 0, 0), -1)

    return Image.fromarray(img)


def solve(img_array):
    """
    Modified solve function to work with numpy array instead of file path
    Returns True if grids were found, False otherwise
    """
    original = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    grid_candidates = find_grid_contour(gray)

    if len(grid_candidates) == 0:
        return False, None, None

    print(f"Found {len(grid_candidates)} potential grids")

    resps, cellss = [], []
    for i, candidate in enumerate(grid_candidates):
        grid_size, cells = find_grid_cells(original, gray, *candidate,
                                            black_threshold=40,
                                            black_ratio=0.25)
        
        print(f"Grid {i+ 1} / {i+1} has size {grid_size}")

        if cells is None:
            continue

        cellss.append(cells)
        grid = np.array([[cell['color_idx'] for cell in row] for row in cells])

        resp = n_queens_find_sol(len(grid), grid)
        resps.append(resp)

    return True, resps, cellss


def open_image_viewer(image_path):
    """
    Open image with default viewer based on OS
    """
    try:
        # Try common Linux image viewers
        viewers = ['eog', 'feh', 'display', 'xdg-open']
        for viewer in viewers:
            try:
                subprocess.Popen([viewer, image_path])
                return
            except FileNotFoundError:
                continue
        print(f"No image viewer found. Image saved at: {image_path}")
    except Exception as e:
        print(f"Error opening image viewer: {e}")


def capture_screen():
    """
    Capture the current screen and return as numpy array
    """
    with mss.mss() as sct:
        # Capture primary monitor
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)

        # Convert to numpy array
        img = np.array(screenshot)
        # Convert BGRA to BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img


def monitor_screen(interval=2.0):
    """
    Continuously monitor the screen for N-Queens grids

    Args:
        interval: Time in seconds between screen captures
    """
    print(f"Starting screen monitor (checking every {interval}s)")
    print("Press Ctrl+C to stop")

    try:
        while True:
            # Capture screen
            screen = capture_screen()

            # Run solve function
            found, resps, cellss = solve(screen)

            # If grids were found, visualize and open
            if found and resps and cellss:
                print("Grid(s) detected! Generating solutions...")

                # Save all solutions
                for idx, (resp, cells) in enumerate(zip(resps, cellss)):
                    for sol_idx, r in enumerate(resp):
                        # Create visualization
                        result_img = visualize_solution(cells, r)

                        # Save to temporary file
                        output_path = data_dir / f"solution_grid{idx}_sol{sol_idx}_{int(time.time())}.png"
                        result_img.save(output_path)
                        print(f"Solution saved: {output_path}")

                        # Open in image viewer
                        open_image_viewer(str(output_path))

                # Wait for user to press Enter before continuing
                input("\nPress Enter to continue monitoring (or Ctrl+C to quit)...")

            # Wait before next capture
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Monitor screen for N-Queens puzzles')
    parser.add_argument('--interval', type=float, default=2.0,
                      help='Seconds between screen captures (default: 2.0)')

    args = parser.parse_args()

    monitor_screen(interval=args.interval)
