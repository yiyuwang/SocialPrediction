#!/usr/bin/env python
"""Backward-compatible wrapper for the renamed preprocessing script."""

from edit_fmap_json import parse_args, update_subject_fieldmaps


if __name__ == "__main__":
    args = parse_args()
    update_subject_fieldmaps(args.bids_dir, args.subject_id)
