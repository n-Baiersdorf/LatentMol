from pathlib import Path
from typing import List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import os  

@dataclass
class DirectoryData:
    """Data class to store directory analysis results."""
    numbers: List[int]
    file_counts: List[int]

class DirectoryAnalyzer:
    """
    A class for analyzing directory structures where subdirectories are named with numbers
    and contain varying numbers of files.
    """

    def __init__(self, root_path: Union[str, Path]):
        """
        Initialize the DirectoryAnalyzer with a root directory path.

        Args:
            root_path: Path to the root directory to analyze

        Raises:
            FileNotFoundError: If the directory doesn't exist
            NotADirectoryError: If the path is not a directory
            PermissionError: If there are insufficient permissions to access the directory
        """
        self.root_path = Path(root_path)
        self._validate_directory()
        self._data: Optional[DirectoryData] = None
        self._current_figure: Optional[plt.Figure] = None

    def _validate_directory(self) -> None:
        """Validate the root directory path."""
        if not self.root_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.root_path}")
        if not self.root_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.root_path}")
        if not os.access(self.root_path, os.R_OK):
            raise PermissionError(f"Insufficient permissions to read directory: {self.root_path}")

    def scan_directory(self) -> None:
        """
        Scan the directory structure and collect data about subdirectories and their contents.

        This method will analyze all immediate subdirectories of the root path that are
        named with numbers and count the files within them.

        Raises:
            ValueError: If no valid numbered subdirectories are found
        """
        numbers = []
        file_counts = []

        for item in self.root_path.iterdir():
            if item.is_dir() and item.name.isdigit():
                number = int(item.name)
                file_count = sum(1 for _ in item.glob('*') if _.is_file())
                numbers.append(number)
                file_counts.append(file_count)

        if not numbers:
            raise ValueError(f"No valid numbered subdirectories found in {self.root_path}")

        # Sort both lists based on numbers
        sorted_pairs = sorted(zip(numbers, file_counts))
        self._data = DirectoryData(
            numbers=[pair[0] for pair in sorted_pairs],
            file_counts=[pair[1] for pair in sorted_pairs]
        )

    def get_data(self) -> DirectoryData:
        """
        Retrieve the collected directory data.

        Returns:
            DirectoryData: Object containing lists of directory numbers and file counts

        Raises:
            RuntimeError: If scan_directory() hasn't been called yet
        """
        if self._data is None:
            raise RuntimeError("No data available. Call scan_directory() first.")
        return self._data

    def plot_distribution(self,
                         log_x: bool = False,
                         log_y: bool = False,
                         x_label: str = "Directory Number",
                         y_label: str = "Number of Files",
                         title: str = "Directory Content Distribution",
                         show_grid: bool = True,
                         marker_style: str = 'o',
                         marker_color: str = 'blue',
                         line_style: str = '-',
                         line_color: str = 'blue',
                         figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot the distribution of files across directories.

        Args:
            log_x: Whether to use logarithmic scale for x-axis
            log_y: Whether to use logarithmic scale for y-axis
            x_label: Label for x-axis
            y_label: Label for y-axis
            title: Plot title
            show_grid: Whether to show grid lines
            marker_style: Style of data points
            marker_color: Color of data points
            line_style: Style of connecting lines
            line_color: Color of connecting lines
            figsize: Figure size as (width, height) in inches

        Raises:
            RuntimeError: If scan_directory() hasn't been called yet
        """
        if self._data is None:
            raise RuntimeError("No data available. Call scan_directory() first.")

        plt.close()  # Close any existing figures
        self._current_figure, ax = plt.subplots(figsize=figsize)

        # Plot data points and lines
        ax.plot(self._data.numbers, self._data.file_counts,
                marker=marker_style, color=marker_color,
                linestyle=line_style)  # Removed incorrect 'linecolor'

        # Set scales
        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')

        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        # Set grid
        ax.grid(show_grid)

        # Adjust layout to prevent label clipping
        plt.tight_layout()

    def save_plot(self,
                  file_path: Union[str, Path],
                  format: str = 'png',
                  dpi: int = 300) -> None:
        """
        Save the current plot to a file.

        Args:
            file_path: Path where to save the plot
            format: File format (e.g., 'png', 'jpg', 'svg')
            dpi: Resolution in dots per inch

        Raises:
            RuntimeError: If no plot has been generated yet
            ValueError: If the file format is not supported
        """
        if self._current_figure is None:
            raise RuntimeError("No plot available. Call plot_distribution() first.")

        supported_formats = {'png', 'jpg', 'jpeg', 'svg', 'pdf'}
        if format.lower() not in supported_formats:
            raise ValueError(f"Unsupported format. Supported formats are: {supported_formats}")

        try:
            save_path = Path(file_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self._current_figure.savefig(save_path, format=format, dpi=dpi)
        except Exception as e:
            raise RuntimeError(f"Failed to save plot: {str(e)}")

if __name__ == "__main__":
    import tempfile
    import shutil

    # Create a temporary directory with a sample structure for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample subdirectories with files
        for i in range(1, 6):
            subdir = Path(temp_dir) / str(i)
            subdir.mkdir()
            # Create some files in each subdirectory
            for j in range(i):
                (subdir / f"file_{j}.txt").touch()

        # Initialize the analyzer with the temporary directory
        analyzer = DirectoryAnalyzer(temp_dir)

        try:
            # Scan the directory structure
            analyzer.scan_directory()

            # Get the collected data
            data = analyzer.get_data()
            print(f"Found {len(data.numbers)} subdirectories")
            print(f"Total files: {sum(data.file_counts)}")

            # Create a customized plot
            analyzer.plot_distribution(
                log_y=True,  # Use log scale for y-axis
                title="File Distribution Across Directories",
                marker_style='o',
                marker_color='red',
                line_style='--',
                line_color='blue',
                show_grid=True
            )

            # Save the plot as a high-resolution PNG
            output_dir = Path.home() / "Downloads"
            output_dir.mkdir(exist_ok=True)
            plot_path = output_dir / "distribution_plot.png"
            analyzer.save_plot(
                plot_path,
                format='png',
                dpi=300
            )
            print(f"Plot saved to {plot_path}")

        except Exception as e:
            print(f"An error occurred: {e}")

            
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = DirectoryAnalyzer("data/test_sort/")

    # Scan the directory structure
    analyzer.scan_directory()

    # Get the collected data
    data = analyzer.get_data()
    print(f"Found {len(data.numbers)} subdirectories")
    print(f"Total files: {sum(data.file_counts)}")
 
    print(data.numbers)

    print("----------------------------------")

    print(data.file_counts)