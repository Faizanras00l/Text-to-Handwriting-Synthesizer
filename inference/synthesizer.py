import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
import numpy as np
import svgwrite
import textwrap

from inference import canvas as drawing
from networks.lstm_layer import rnn


class Hand(object):

    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        self.nn = rnn(
            log_dir=os.path.join(root_dir, 'results', 'logs'),
            checkpoint_dir=os.path.join(root_dir, 'resources', 'checkpoints'),
            prediction_dir=os.path.join(root_dir, 'results', 'predictions'),
            learning_rates=[.0001, .00005, .00002],
            batch_sizes=[32, 64, 64],
            patiences=[1500, 1000, 500],
            beta1_decays=[.9, .9, .9],
            validation_batch_size=32,
            optimizer='rms',
            num_training_steps=100000,
            warm_start_init_step=17900,
            regularization_constant=0.0,
            keep_prob=1.0,
            enable_parameter_averaging=False,
            min_steps_to_checkpoint=2000,
            log_interval=20,
            logging_level=logging.CRITICAL,
            grad_clip=10,
            lstm_size=400,
            output_mixture_components=20,
            attention_mixture_components=10
        )
        self.nn.restore()

    def write(
        self,
        filename,
        lines,
        biases=None,
        styles=None,
        stroke_colors=None,
        stroke_widths=None,
        # ================= NEW PARAMETERS FOR LINED PAPER =================
        page_width=1860,          # Page width in pixels (A4-like)
        page_height=3508,         # Page height in pixels (A4-like)
        margins=None,             # Dict with 'left', 'right', 'top', 'bottom' margins
        ruled=False,              # True = lined paper, False = blank paper (default)
        lines_per_page=22,        # Number of lines per page
        line_gap=125              # Fixed line spacing in pixels
        # ===================================================================
    ):
        """
        Generate handwriting from text.
        
        Args:
            filename: Output SVG filename
            lines: Text to synthesize (string or list of strings)
            biases: List of bias values (0.5-1.0) for each line
            styles: List of style indices (0-12) for each line
            stroke_colors: List of stroke colors for each line
            stroke_widths: List of stroke widths for each line
            page_width: Page width in pixels (default: 1860)
            page_height: Page height in pixels (default: 3508)
            margins: Dict with margins (default: left=150, right=150, top=250, bottom=250)
            ruled: If True, draw lined paper; if False, draw blank paper (default: False)
        
        Returns:
            List of generated SVG filenames
        """
        
        # Set default margins if not provided
        if margins is None:
            margins = {"left": 150, "right": 150, "top": 250, "bottom": 250}

        # =====================================================================
        # LINED PAPER MODE - Uses fixed layout with 25 lines per page
        # =====================================================================
        if ruled:
            # ================= LINED PAPER LAYOUT SETTINGS =================
            # These values control the lined paper appearance
            # You can easily modify these to customize the output
            
            # Use passed-in values (no calculation!)
            LINES_PER_PAGE = lines_per_page
            LINE_GAP = line_gap
            
            SCALE = 2.4                  # Text scale factor (larger = bigger text)
            
            # Character limit per line (controls text wrapping)
            usable_width = page_width - margins["left"] - margins["right"]
            char_limit = int(usable_width / 28)  # ~55 chars for default width (RNN sweet spot)
            # ===============================================================

            # Wrap text to fit within character limit
            # PRESERVES intentional line breaks (paragraphs stay separate)
            final_lines = []
            if isinstance(lines, str):
                # Single string: split by newlines, wrap each paragraph
                paragraphs = lines.split('\n')
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    if len(para) <= char_limit:
                        final_lines.append(para)
                    else:
                        # Wrap long paragraph at word boundaries
                        words = para.split()
                        current_line = []
                        current_length = 0
                        for word in words:
                            space_needed = current_length + (1 if current_line else 0) + len(word)
                            if space_needed > char_limit and current_line:
                                final_lines.append(' '.join(current_line))
                                current_line = [word]
                                current_length = len(word)
                            else:
                                current_line.append(word)
                                current_length = space_needed
                        if current_line:
                            final_lines.append(' '.join(current_line))
            else:
                # List of lines: wrap each line individually (preserve breaks)
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if len(line) <= char_limit:
                        final_lines.append(line)
                    else:
                        # Wrap long line at word boundaries
                        words = line.split()
                        current_line = []
                        current_length = 0
                        for word in words:
                            space_needed = current_length + (1 if current_line else 0) + len(word)
                            if space_needed > char_limit and current_line:
                                final_lines.append(' '.join(current_line))
                                current_line = [word]
                                current_length = len(word)
                            else:
                                current_line.append(word)
                                current_length = space_needed
                        if current_line:
                            final_lines.append(' '.join(current_line))

            print(f"Layout: {len(final_lines)} lines total | {LINES_PER_PAGE} lines/page")

            total_lines = len(final_lines)
            
            # Extend biases/styles to match line count
            if biases is None: 
                biases = [1.0] * total_lines
            elif len(biases) < total_lines: 
                biases += [biases[-1]] * (total_lines - len(biases))
            
            if styles is None: 
                styles = [0] * total_lines
            elif len(styles) < total_lines: 
                styles += [styles[-1]] * (total_lines - len(styles))

            # Generate pages
            generated_files = []
            for page_num, i in enumerate(range(0, total_lines, LINES_PER_PAGE)):
                chunk_lines = final_lines[i : i + LINES_PER_PAGE]
                chunk_biases = biases[i : i + LINES_PER_PAGE]
                chunk_styles = styles[i : i + LINES_PER_PAGE]
                
                page_filename = filename.replace(".svg", f"_p{page_num+1}.svg")
                
                strokes = self._sample(chunk_lines, biases=chunk_biases, styles=chunk_styles)
                
                self._draw_lined(
                    strokes, chunk_lines, page_filename,
                    page_width, page_height, margins,
                    LINE_GAP, SCALE
                )
                generated_files.append(page_filename)
                
            return generated_files
        
        # =====================================================================
        # BLANK PAPER MODE - Original behavior (single page, dynamic sizing)
        # =====================================================================
        else:
            valid_char_set = set(drawing.alphabet)
            for line_num, line in enumerate(lines):
                if len(line) > 75:
                    raise ValueError(
                        (
                            "Each line must be at most 75 characters. "
                            "Line {} contains {}"
                        ).format(line_num, len(line))
                    )

                for char in line:
                    if char not in valid_char_set:
                        raise ValueError(
                            (
                                "Invalid character {} detected in line {}. "
                                "Valid character set is {}"
                            ).format(char, line_num, valid_char_set)
                        )

            strokes = self._sample(lines, biases=biases, styles=styles)
            self._draw_blank(strokes, lines, filename, stroke_colors=stroke_colors, stroke_widths=stroke_widths)
            return [filename]

    def _sample(self, lines, biases=None, styles=None):
        """Sample strokes from the RNN model."""
        num_samples = len(lines)
        max_tsteps = 60 * max(len(l) for l in lines)
        biases = biases if biases is not None else [0.5] * num_samples

        x_prime = np.zeros([num_samples, 1200, 3])
        x_prime_len = np.zeros([num_samples])
        chars = np.zeros([num_samples, 120])
        chars_len = np.zeros([num_samples])

        if styles is not None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(script_dir)
            for i, (cs, style) in enumerate(zip(lines, styles)):
                x_p = np.load(os.path.join(root_dir, 'resources', 'styles', 'style-{}-strokes.npy'.format(style)))
                c_p = np.load(os.path.join(root_dir, 'resources', 'styles', 'style-{}-chars.npy'.format(style))).tostring().decode('utf-8')

                c_p = str(c_p) + " " + cs
                c_p = drawing.encode_ascii(c_p)
                c_p = np.array(c_p)

                x_prime[i, :len(x_p), :] = x_p
                x_prime_len[i] = len(x_p)
                chars[i, :len(c_p)] = c_p
                chars_len[i] = len(c_p)

        else:
            for i in range(num_samples):
                encoded = drawing.encode_ascii(lines[i])
                chars[i, :len(encoded)] = encoded
                chars_len[i] = len(encoded)

        [samples] = self.nn.session.run(
            [self.nn.sampled_sequence],
            feed_dict={
                self.nn.prime: styles is not None,
                self.nn.x_prime: x_prime,
                self.nn.x_prime_len: x_prime_len,
                self.nn.num_samples: num_samples,
                self.nn.sample_tsteps: max_tsteps,
                self.nn.c: chars,
                self.nn.c_len: chars_len,
                self.nn.bias: biases
            }
        )
        samples = [sample[~np.all(sample == 0.0, axis=1)] for sample in samples]
        return samples

    # =========================================================================
    # LINED PAPER DRAWING METHOD
    # =========================================================================
    def _draw_lined(self, strokes, lines, filename, width, height, margins, line_gap, scale):
        """
        Draw handwriting on lined (ruled) paper.
        
        This method draws notebook-style ruled paper with:
        - Vertical margin lines (gray)
        - Double red header line at top
        - Blue horizontal ruled lines
        """
        LEFT, RIGHT = margins["left"], width - margins["right"]
        TOP, BOTTOM = margins["top"], height - margins["bottom"]
        
        WRITING_AREA_WIDTH = width - margins["left"] - margins["right"]

        # ================= LINED PAPER STYLE SETTINGS =================
        # Easily customizable visual settings for the ruled paper
        
        BASELINE_OFFSET = -15        # Vertical offset for text on lines
        DEFAULT_X_STRETCH = 1.5     # Horizontal word spacing (1.5 = wider gaps)
        
        # Line colors (use HTML color codes or names)
        MARGIN_LINE_COLOR = "#DBDBDB"    # Gray vertical margin lines
        HEADER_LINE_COLOR = "#FFB0B0"    # Pink/red header lines
        RULED_LINE_COLOR = "#7BA3C4"     # Blue horizontal lines (darker, more visible)
        
        # Line widths
        MARGIN_LINE_WIDTH = 2
        HEADER_LINE_WIDTH = 2
        RULED_LINE_WIDTH = 1.5           # Thicker for better visibility
        
        # Stroke settings
        TEXT_STROKE_COLOR = "black"
        TEXT_STROKE_WIDTH = 2.4
        # ==============================================================

        dwg = svgwrite.Drawing(filename, size=(width, height), viewBox=f"0 0 {width} {height}")
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))

        # ----- Draw Ruled Lines -----
        
        # Vertical margin lines
        dwg.add(dwg.line(start=(LEFT-20, 0), end=(LEFT-20, height), 
                        stroke=MARGIN_LINE_COLOR, stroke_width=MARGIN_LINE_WIDTH))
        dwg.add(dwg.line(start=(RIGHT+20, 0), end=(RIGHT+20, height), 
                        stroke=MARGIN_LINE_COLOR, stroke_width=MARGIN_LINE_WIDTH))
        
        # Double red header line at top
        dwg.add(dwg.line(start=(0, TOP), end=(width, TOP), 
                        stroke=HEADER_LINE_COLOR, stroke_width=HEADER_LINE_WIDTH))
        dwg.add(dwg.line(start=(0, TOP + 6), end=(width, TOP + 6), 
                        stroke=HEADER_LINE_COLOR, stroke_width=HEADER_LINE_WIDTH))
        
        # Blue horizontal ruled lines
        y = TOP + line_gap
        while y < height - margins["bottom"] + 10:  # +10 ensures last line is drawn
            dwg.add(dwg.line(start=(0, y), end=(width, y), 
                           stroke=RULED_LINE_COLOR, stroke_width=RULED_LINE_WIDTH))
            y += line_gap

        # ----- Draw Handwriting -----
        
        # Start writing on the first BLUE line (after the red header)
        y_cursor = TOP + line_gap

        for i, (offsets, text) in enumerate(zip(strokes, lines)):
            if not text:
                y_cursor += line_gap
                continue

            offsets = offsets.copy()
            offsets[:, :2] *= scale
            coords = drawing.offsets_to_coords(offsets)

            if len(coords) == 0: 
                y_cursor += line_gap
                continue
            
            try:
                denoised = drawing.denoise(coords)
                if len(denoised) > 2: 
                    coords = denoised
            except: 
                pass
            
            try:
                if len(coords) > 2: 
                    coords[:, :2] = drawing.align(coords[:, :2])
            except: 
                pass

            coords[:, 1] *= -1
            coords[:, 0] -= coords[:, 0].min()
            
            # === Text Justification & Stretching ===
            current_width = coords[:, 0].max()
            is_last_line = (i == len(lines) - 1)
            
            projected_width = current_width * DEFAULT_X_STRETCH
            fill_ratio = projected_width / WRITING_AREA_WIDTH
            
            # If line is > 75% full, stretch it to fill the line
            if fill_ratio > 0.75 and not is_last_line:
                final_stretch = WRITING_AREA_WIDTH / current_width
                if final_stretch > 1.7: 
                    final_stretch = 1.7
                coords[:, 0] *= final_stretch
            else:
                # Use default wide spacing
                coords[:, 0] *= DEFAULT_X_STRETCH
            
            coords[:, 0] += LEFT
            coords[:, 1] += y_cursor + BASELINE_OFFSET

            # Build SVG path
            path = ""
            prev = 1.0
            for x, y, eos in coords:
                path += f"{'M' if prev else 'L'}{x},{y} "
                prev = eos

            dwg.add(svgwrite.path.Path(path).stroke(TEXT_STROKE_COLOR, width=TEXT_STROKE_WIDTH, linecap="round").fill("none"))
            y_cursor += line_gap

        dwg.save()
        print(f"   ✅ Saved: {filename}")

    # =========================================================================
    # BLANK PAPER DRAWING METHOD (Original behavior)
    # =========================================================================
    def _draw_blank(self, strokes, lines, filename, stroke_colors=None, stroke_widths=None):
        """
        Draw handwriting on blank white paper.
        
        This is the original drawing method with dynamic canvas sizing.
        """
        stroke_colors = stroke_colors or ['black'] * len(lines)
        stroke_widths = stroke_widths or [2] * len(lines)

        # ================= BLANK PAPER LAYOUT SETTINGS =================
        # Easily customizable settings for blank paper
        
        LINE_HEIGHT = 80         # Vertical spacing between lines
        VIEW_WIDTH = 1200        # Canvas width
        TOP_MARGIN = 40          # Top padding
        BOTTOM_MARGIN = 40       # Bottom padding
        LEFT_MARGIN = 50         # Left margin for overflow cases
        # ===============================================================
        
        # Calculate dynamic canvas height based on line count
        num_lines = len([l for l in lines if l])
        content_height = num_lines * LINE_HEIGHT
        view_height = content_height + TOP_MARGIN + BOTTOM_MARGIN
        
        if num_lines > 10:
            import warnings
            warnings.warn(f"Generating {num_lines} lines. Quality may vary for very long documents.")

        dwg = svgwrite.Drawing(filename=filename)
        dwg.viewbox(width=VIEW_WIDTH, height=view_height)
        dwg.add(dwg.rect(insert=(0, 0), size=(VIEW_WIDTH, view_height), fill='white'))

        line_index = 0
        
        for offsets, line, color, width in zip(strokes, lines, stroke_colors, stroke_widths):
            if not line:
                continue

            offsets_copy = offsets.copy()
            offsets_copy[:, :2] *= 1.5
            line_strokes = drawing.offsets_to_coords(offsets_copy)
            line_strokes = drawing.denoise(line_strokes)
            
            # Extra smoothing for end-of-line distortion
            smooth_threshold = int(len(line_strokes) * 0.75)
            if smooth_threshold > 10 and len(line_strokes) > 20:
                ending_portion = line_strokes[smooth_threshold:].copy()
                ending_portion = drawing.denoise(ending_portion)
                line_strokes[smooth_threshold:] = ending_portion
            
            line_strokes[:, :2] = drawing.align(line_strokes[:, :2])

            base_y = TOP_MARGIN + (line_index * LINE_HEIGHT)
            
            line_strokes[:, 1] *= -1
            min_x = line_strokes[:, 0].min()
            min_y = line_strokes[:, 1].min()
            line_strokes[:, 0] -= min_x
            line_strokes[:, 1] -= min_y
            
            stroke_width_val = line_strokes[:, 0].max()
            if stroke_width_val < VIEW_WIDTH - (2 * LEFT_MARGIN):
                x_offset = (VIEW_WIDTH - stroke_width_val) / 2
            else:
                x_offset = LEFT_MARGIN
                
            line_strokes[:, 0] += x_offset
            line_strokes[:, 1] += base_y

            prev_eos = 1.0
            p = "M{},{} ".format(0, 0)
            for x, y, eos in zip(*line_strokes.T):
                p += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x, y)
                prev_eos = eos
            
            path = svgwrite.path.Path(p)
            path = path.stroke(color=color, width=width, linecap='round').fill("none")
            dwg.add(path)
            
            line_index += 1

        dwg.save()
        print(f"   ✅ Saved: {filename}")
