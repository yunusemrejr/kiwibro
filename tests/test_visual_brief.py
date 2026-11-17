"""Only unambiguous requested fields are bound into the visual component route."""

import pytest

from kiwibro.visual_brief import hinge_bindings


def test_detailed_brief_binds_roles_in_list_order() -> None:
    message = (
        "Save blower-access.htm, a single offline HTML inspection chamber. Show a filter bay beneath a service lid. "
        "Use perspective and true CSS 3D motion so the lid opens toward the rear on its top-edge hinge with a fixed base. "
        "Give the scene, base, and lid the IDs housing-view, fixed-mount, and access-cover. "
        "Start closed. Use a native button with ID inspect that opens and closes the lid, keeps keyboard focus, "
        "and updates aria-expanded. The button should fit mobile at 360px and offer at least a 40 × 40 px target. "
        "Each pose stays in place until activation. Honor reduced motion. Check the saved render with look_at_page."
    )
    assert hinge_bindings(message) == {
        "filename": "blower-access.htm", "edge": "top", "direction": "away", "scene_id": "housing-view",
        "frame_id": "fixed-mount", "leaf_id": "access-cover", "toggle_id": "inspect", "part": "service lid",
        "detail": "A filter bay is behind the service lid.",
    }


def test_explicit_labels_and_elements_are_preserved_as_written() -> None:
    message = (
        'Could you save "Pump Bay.html" as a self-contained HTML file? '
        "Show a service bay beneath a rear lid. The hatch hinges at its right edge on a fixed frame. "
        "Swing its free edge toward the viewer. Use IDs pump-leaf for the moving part, "
        "pump-base for the fixed frame, and pump-toggle for the native button named “Inspect pump”. "
        "Toggle the hidden detail pump-detail. Show “Pump at rest.” then “Pump visible.” "
        "in the visible status pump-status. Set the page title to “Pump Inspection” and the scene ID to pump-stage. "
        "Name the moving leaf “Access leaf”. The revealed detail should read “The pump is accessible.” "
        "Open the hinged leaf to 63°."
    )
    assert hinge_bindings(message) == {
        "filename": "Pump Bay.html", "edge": "right", "direction": "toward", "angle": "63", "title": "Pump Inspection",
        "part": "Access leaf", "detail": "The pump is accessible.", "button_label": "Inspect pump",
        "closed_label": "Pump at rest.", "open_label": "Pump visible.", "leaf_id": "pump-leaf", "frame_id": "pump-base",
        "toggle_id": "pump-toggle", "detail_id": "pump-detail", "status_id": "pump-status", "scene_id": "pump-stage",
    }


@pytest.mark.parametrize("edge", ["top", "bottom", "left", "right"])
def test_one_edge_and_explicit_quoted_id_can_be_bound(edge: str) -> None:
    result = hinge_bindings(f"Build `compact.html` with a {edge} hinge. Give the button ID “toggle-a”.")
    assert result == {"filename": "compact.html", "edge": edge, "toggle_id": "toggle-a"}


def test_text_in_quotes_does_not_supply_an_edge_or_id() -> None:
    assert hinge_bindings('Create panel.html with a label named "top hinge".') is None
    result = hinge_bindings(
        'Create panel.html with a right hinge. Use the button label "Next: button with ID wrong or a left hinge."'
    )
    assert result == {
        "filename": "panel.html", "edge": "right", "button_label": "Next: button with ID wrong or a left hinge.",
    }


@pytest.mark.parametrize("message", [
    "Why does test.html need a top-edge hinge?",
    "Edit old.html so it uses a top-edge hinge.",
    "Create old.html with a top hinge. Update the existing page as well.",
    "Read the existing old.html. Save old.html with a top hinge.",
    "Save over old.html with a top hinge.",
    "Create ../outside.html with a top hinge.",
    'Create "../outside.html" with a top hinge.',
    "Create https://example.test/remote.html with a top hinge.",
    "Create one.html and two.htm with a top hinge.",
    "Create one.html with a top or bottom hinge.",
    "Create one.html with a top/bottom-edge hinge.",
    "Create one.html without a top hinge.",
    "Create one.html with a top hinge and a bottom hinge.",
    "Create one.html with a top hinge for two lids.",
    "Create one.html with a top hinge and a second independently moving door.",
    "Create one.html with a top hinge. Use the following CSS.",
    "Create one.html with a top hinge. Use WebGL.",
    "Create one.html with a top hinge, then publish it to the website.",
    "Create one.html with a top hinge. Give the scene, base and lid IDs root, cover.",
    "Create one.html with a top hinge. Give the scene ID seat and the button ID seat.",
    "Create one.html with a top hinge. Use the button ID yes. Give the button ID no.",
    "Create one.html with a top hinge. Rotate the panel to 120 degrees.",
    "Create one.html with a top hinge. Open toward the rear and toward the camera.",
    "Create one.html with a top hinge. Show a mail slot under a cover. Reveal a vent behind a service lid.",
])
def test_ambiguous_or_unsupported_briefs_defer(message: str) -> None:
    assert hinge_bindings(message) is None
