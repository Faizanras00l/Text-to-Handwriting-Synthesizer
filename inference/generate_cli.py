import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from inference.synthesizer import Hand
# =============================================================================
# PAGE SETTINGS - Easily customizable page dimensions and margins
# =============================================================================
# These settings apply to LINED PAPER mode only
# Blank paper uses dynamic sizing based on content

# LINE CONFIGURATION (change LINES_PER_PAGE to adjust page)
LINES_PER_PAGE = 17        # Number of lines per page (changed from 22)
LINE_GAP = 125             # Fixed spacing between lines (DO NOT CHANGE)

# Page width
PAGE_WIDTH = 1860          # Page width in pixels (A4-like proportion)

# Margins (in pixels)
LEFT_MARGIN = 150          # Left margin
RIGHT_MARGIN = 150         # Right margin
TOP_MARGIN = 250           # Top margin
BOTTOM_MARGIN = 250        # Bottom margin

# Auto-calculated page height (maintains constant line spacing)
PAGE_HEIGHT = (LINES_PER_PAGE * LINE_GAP) + TOP_MARGIN + BOTTOM_MARGIN
# For 17 lines: (17 × 125) + 250 + 250 = 2625 pixels

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

OUTPUT_FOLDER = os.path.join("results", "output")   # All outputs will be saved here

# =============================================================================
# TEXT WRAPPING SETTINGS
# =============================================================================

MAX_CHARS_PER_LINE = 55    # Maximum characters per line (RNN sweet spot: 40-55)

# =============================================================================


def ensure_output_folder():
    """Create output folder if it doesn't exist."""
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"📁 Created output folder: {os.path.abspath(OUTPUT_FOLDER)}")
    return OUTPUT_FOLDER


def smart_wrap(text, max_chars=MAX_CHARS_PER_LINE):
    """
    Professional text wrapping that:
    - Wraps at word boundaries (never breaks mid-word)
    - Handles words longer than max_chars by splitting them
    - Produces clean, uniform line lengths
    - Gives the RNN model optimal line lengths for best quality
    
    Args:
        text: Single line/paragraph of text to wrap
        max_chars: Maximum characters per line (default: 55)
    
    Returns:
        List of wrapped lines
    """
    if not text or not text.strip():
        return []
    
    words = text.split()
    if not words:
        return []
    
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word)
        
        # Handle words longer than max_chars (rare but possible)
        if word_length > max_chars:
            # First, save any current line in progress
            if current_line:
                lines.append(' '.join(current_line))
                current_line = []
                current_length = 0
            
            # Break the long word into chunks
            while len(word) > max_chars:
                lines.append(word[:max_chars])
                word = word[max_chars:]
            
            # Remaining part becomes start of new line
            if word:
                current_line = [word]
                current_length = len(word)
            continue
        
        # Check if adding this word would exceed the limit
        # +1 for the space between words
        space_needed = current_length + (1 if current_line else 0) + word_length
        
        if space_needed > max_chars and current_line:
            # Current line is full, save it and start new line
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = word_length
        else:
            # Add word to current line
            current_line.append(word)
            current_length = space_needed
    
    # Don't forget the last line
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


def convert_svg_to_pdf(svg_files, pdf_path):
    """
    Convert SVG file(s) to a single PDF.
    Requires: pip install svglib reportlab
    """
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPDF
        from reportlab.pdfgen import canvas
        
        print("\n📄 Converting to PDF...")
        c = canvas.Canvas(pdf_path, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
        
        for svg_file in svg_files:
            drawing = svg2rlg(svg_file)
            renderPDF.draw(drawing, c, 0, 0)
            c.showPage()
        
        c.save()
        print(f"✅ PDF saved: {os.path.abspath(pdf_path)}")
        return True
        
    except ImportError:
        print("\n⚠️ PDF conversion failed: Install with 'pip install svglib reportlab'")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate handwriting from text.")
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--style", type=int, help="Style index (0-12)")
    parser.add_argument("--bias", type=float, help="Bias (cleanliness), 0.5-1.0")
    parser.add_argument("--output", type=str, help="Output filename (without extension)")
    parser.add_argument("--paper", type=str, choices=["blank", "lined"], help="Paper type: blank or lined")
    parser.add_argument("--format", type=str, choices=["svg", "pdf"], help="Output format: svg or pdf")
    
    args = parser.parse_args()
    
    # Ensure output folder exists
    ensure_output_folder()
    
    # Interactive mode if arguments not provided
    print("=" * 60)
    print("   HANDWRITING SYNTHESIS - Interactive Mode")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # 1. GET TEXT INPUT
    # -------------------------------------------------------------------------
    text = args.text
    if not text:
        try:
            print("\nEnter the text you want to convert to handwriting:")
            print("(You can use \\n for new lines)")
            text = input("Text: ").strip()
        except EOFError:
            print("No input provided.")
            return

    if not text:
        print("No text provided. Exiting.")
        return

    # -------------------------------------------------------------------------
    # 2. GET STYLE (0-12)
    # -------------------------------------------------------------------------
    style = args.style
    if style is None:
        try:
            print("\n" + "-" * 60)
            print("Available handwriting styles: 0 to 12")
            print("(Each style represents a different handwriting pattern)")
            style_input = input("Choose style [default: 0]: ").strip()
            style = int(style_input) if style_input else 0
            if style < 0 or style > 12:
                print("Invalid style. Using default (0)")
                style = 0
        except (ValueError, EOFError):
            print("Invalid input. Using default style (0)")
            style = 0

    # -------------------------------------------------------------------------
    # 3. GET BIAS (0.1 - 1.5)
    # -------------------------------------------------------------------------
    bias = args.bias
    if bias is None:
        try:
            print("\n" + "-" * 60)
            print("Bias controls handwriting cleanliness & consistency:")
            print("  0.5  = Artistic (most variation, natural imperfections)")
            print("  0.65 = Casual (relaxed handwriting feel)")
            print("  0.75 = Balanced (good mix of natural & clean)")
            print("  0.85 = Neat (clean with slight natural variation)")
            print("  1.0  = Professional (clean & uniform) ⭐ RECOMMENDED")
            print("  1.2  = Ultra-clean (very consistent, minimal variation)")
            print("  1.5  = Maximum precision (near-mechanical consistency)")
            bias_input = input("Choose bias [default: 1.0]: ").strip()
            bias = float(bias_input) if bias_input else 1.0
            if bias < 0.1 or bias > 1.5:
                print("Bias out of range (0.1-1.5). Using default (1.0)")
                bias = 1.0
        except (ValueError, EOFError):
            print("Invalid input. Using default bias (1.0)")
            bias = 1.0

    # -------------------------------------------------------------------------
    # 4. GET PAPER TYPE (1 = blank, 2 = lined)
    # -------------------------------------------------------------------------
    paper_type = args.paper
    if paper_type is None:
        try:
            print("\n" + "-" * 60)
            print("Select paper type:")
            print("  1. Blank page (clean white paper)")
            print("  2. Lined page (notebook-style ruled paper)")
            paper_input = input("Enter choice [1 or 2, default: 1]: ").strip()
            
            if paper_input == "" or paper_input == "1":
                paper_type = "blank"
            elif paper_input == "2":
                paper_type = "lined"
            else:
                print("Invalid choice. Using default (blank)")
                paper_type = "blank"
        except EOFError:
            paper_type = "blank"
    
    # Convert to boolean for the Hand.write() method
    ruled = (paper_type == "lined")

    # -------------------------------------------------------------------------
    # 5. GET OUTPUT FORMAT (1 = SVG, 2 = PDF, 3 = JPG)
    # -------------------------------------------------------------------------
    output_format = args.format
    if output_format is None:
        try:
            print("\n" + "-" * 60)
            print("Select output format:")
            print("  1. SVG (default - editable vector graphics)")
            print("  2. PDF (printable document)")
            format_input = input("Enter choice [1 or 2, default: 1]: ").strip()
            
            if format_input == "" or format_input == "1":
                output_format = "svg"
            elif format_input == "2":
                output_format = "pdf"
            else:
                print("Invalid choice. Using default (SVG)")
                output_format = "svg"
        except EOFError:
            output_format = "svg"

    # -------------------------------------------------------------------------
    # 6. GET OUTPUT FILENAME
    # -------------------------------------------------------------------------
    output = args.output
    if not output:
        try:
            print("\n" + "-" * 60)
            output_input = input("Output filename (without extension) [default: output]: ").strip()
            output = output_input if output_input else "output"
        except EOFError:
            output = "output"
    
    # Remove any extension if user accidentally added one
    if output.endswith(('.svg', '.pdf', '.jpg', '.jpeg', '.png')):
        output = os.path.splitext(output)[0]
    
    # Build full output path with folder
    svg_output = os.path.join(OUTPUT_FOLDER, f"{output}.svg")

    # -------------------------------------------------------------------------
    # PROCESS TEXT
    # -------------------------------------------------------------------------
    
    # Handle newlines in text
    lines = text.replace('\\n', '\n').split('\n')
    lines = [l.strip() for l in lines if l.strip()]
    
    # For blank paper mode: wrap long lines to prevent distortion
    if not ruled:
        wrapped_lines = []
        for line in lines:
            if len(line) > MAX_CHARS_PER_LINE:
                wrapped_lines.extend(smart_wrap(line))
            else:
                wrapped_lines.append(line)
        lines = wrapped_lines

    if not lines:
        print("No valid text lines.")
        return

    # -------------------------------------------------------------------------
    # DISPLAY SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📝 GENERATION SETTINGS")
    print("=" * 60)
    print(f"   Text: {' / '.join(lines[:3])}{'...' if len(lines) > 3 else ''}")
    print(f"   Lines: {len(lines)}")
    print(f"   Style: {style}")
    print(f"   Bias: {bias}")
    print(f"   Paper: {paper_type.upper()}")
    print(f"   Format: {output_format.upper()}")
    print(f"   Output: {OUTPUT_FOLDER}/{output}.{output_format}")
    print("=" * 60)
    print("\nInitializing model (this may take a moment)...")
    
    # -------------------------------------------------------------------------
    # GENERATE HANDWRITING
    # -------------------------------------------------------------------------
    hand = Hand()
    
    print(f"Generating handwriting for {len(lines)} line(s)...")
    
    # Expand scalar args to lists matching lines
    biases = [bias] * len(lines)
    styles = [style] * len(lines)
    
    # Prepare margins dict
    margins = {
        "left": LEFT_MARGIN,
        "right": RIGHT_MARGIN,
        "top": TOP_MARGIN,
        "bottom": BOTTOM_MARGIN
    }
    
    # Generate handwriting (always creates SVG first)
    generated_files = hand.write(
        filename=svg_output,
        lines=lines,
        biases=biases,
        styles=styles,
        page_width=PAGE_WIDTH,
        page_height=PAGE_HEIGHT,
        margins=margins,
        ruled=ruled,
        lines_per_page=LINES_PER_PAGE,
        line_gap=LINE_GAP
    )
    
    # -------------------------------------------------------------------------
    # CONVERT TO REQUESTED FORMAT
    # -------------------------------------------------------------------------
    final_files = generated_files  # Default: SVG files
    
    if output_format == "pdf":
        pdf_path = os.path.join(OUTPUT_FOLDER, f"{output}.pdf")
        if convert_svg_to_pdf(generated_files, pdf_path):
            final_files = [pdf_path]
            # Optionally delete SVG files after PDF conversion
            # for f in generated_files:
            #     os.remove(f)
    

    
    # -------------------------------------------------------------------------
    # OUTPUT RESULTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("✅ SUCCESS! Handwriting generated:")
    for f in final_files:
        print(f"   📄 {os.path.abspath(f)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
