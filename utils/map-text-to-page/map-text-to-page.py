import argparse
import xml.etree.ElementTree as ET
import editdistance
import numpy as np


def parse_pagexml(file_path):
    """
    Parse the PageXML file and extract text lines.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    namespace = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    text_lines = []

    for text_line in root.findall(".//ns:TextLine", namespace):
        line_id = text_line.get("id")
        text_equiv = text_line.find(".//ns:TextEquiv/ns:Unicode", namespace)
        if text_equiv is not None:
            text_lines.append((line_id, text_equiv.text))

    return tree, root, namespace, text_lines


def read_ground_truth(file_path):
    """
    Read the ground truth text file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def map_textlines(pagexml_lines, ground_truth_lines):
    """
    Map ground truth lines to PageXML lines to minimize the total combined Levenshtein distance.
    """
    num_gt = len(ground_truth_lines)
    num_pagexml = len(pagexml_lines)

    # Create a cost matrix for Levenshtein distances
    cost_matrix = np.full((num_gt, num_pagexml), float("inf"))
    for i, gt_line in enumerate(ground_truth_lines):
        for j, (line_id, page_line) in enumerate(pagexml_lines):
            cost_matrix[i, j] = editdistance.eval(gt_line, page_line)

    # Solve the assignment problem using Hungarian algorithm
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create the mapping
    mapping = []
    for i, j in zip(row_ind, col_ind):
        gt_line = ground_truth_lines[i]
        line_id, page_line = pagexml_lines[j]
        distance = cost_matrix[i, j]
        mapping.append((gt_line, line_id, page_line, distance))

    # Add unmatched ground truth lines
    matched_gt_indices = set(row_ind)
    for i, gt_line in enumerate(ground_truth_lines):
        if i not in matched_gt_indices:
            mapping.append((gt_line, None, None, None))

    return mapping


def update_pagexml_with_matches(tree, root, namespace, mapping):
    """
    Update the PageXML file with the best matches from the ground truth.
    """
    for gt_line, line_id, page_line, _ in mapping:
        if line_id is not None:
            text_line = root.find(f".//ns:TextLine[@id='{line_id}']", namespace)
            if text_line is not None:
                text_equiv = text_line.find(".//ns:TextEquiv/ns:Unicode", namespace)
                if text_equiv is not None:
                    text_equiv.text = gt_line  # Replace with the best match

    return tree


def main():
    parser = argparse.ArgumentParser(description="Map ground truth text to PageXML text lines.")
    parser.add_argument("--pagexml_path", help="Path to the PageXML file.")
    parser.add_argument("--ground_truth_path", help="Path to the ground truth text file.")
    parser.add_argument("--output_path", help="Path to the output file.")
    args = parser.parse_args()

    # Parse PageXML and ground truth
    tree, root, namespace, pagexml_lines = parse_pagexml(args.pagexml_path)
    ground_truth_lines = read_ground_truth(args.ground_truth_path)

    # Map text lines
    mapping = map_textlines(pagexml_lines, ground_truth_lines)
    # Count lines without match
    count_lines_without_match = sum(1 for _, line_id, _, _ in mapping if line_id is None)

    # Update PageXML with matches
    updated_tree = update_pagexml_with_matches(tree, root, namespace, mapping)

    # Write the updated PageXML to the output file
    updated_tree.write(args.output_path, encoding="utf-8", xml_declaration=True)

    # Write results to output
    total_distance = 0
    for gt_line, line_id, page_line, distance in mapping:
        if distance is not None and distance > 0:  # Ensure distance is not None
            print(f"GT: {gt_line}\n")
            print(f"Matched Line ID: {line_id}\n")
            print(f"Matched Text: {page_line}\n")
            print(f"Levenshtein Distance: {distance}\n")
            print("\n")
            total_distance += distance

    print(f'lines without match: {count_lines_without_match}\n')

    print(f'Total Levenshtein Distance: {total_distance}\n')


if __name__ == "__main__":
    main()