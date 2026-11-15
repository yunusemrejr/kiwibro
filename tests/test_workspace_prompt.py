"""Message 0 of a chat in a temporary folder: the same text in every such chat, so the model's cached prefix is shared."""

from kiwibro import config, prompts, workspaces
from kiwibro.agent import AgentManager
from kiwibro.mind import render


def _message_zero(folder) -> str:
    return AgentManager._system_prompt({"workspace": str(folder)})


def test_two_temporary_chats_share_message_zero():
    first = _message_zero(workspaces.temp_folder_of("c_aaaaaaaa11111111"))
    second = _message_zero(workspaces.temp_folder_of("c_bbbbbbbb22222222"))
    assert first == second
    assert "temporary folder that belongs to this chat" in first
    assert str(workspaces.temp_root()) not in first


def test_a_picked_folder_is_named(tmp_path):
    text = _message_zero(tmp_path)
    assert f"work inside the folder `{config.display_path(str(tmp_path))}`" in text
    assert "temporary folder" not in text


def test_kiwi_mind_is_told_about_the_temporary_folder_without_its_path(tmp_path):
    temporary = [{"role": "system", "content": _message_zero(workspaces.temp_folder_of("c_aaaaaaaa11111111"))}]
    assert render._system_facts(temporary)[0] == "this chat's temporary folder"
    picked = [{"role": "system", "content": _message_zero(tmp_path)}]
    assert render._system_facts(picked)[0] == config.display_path(str(tmp_path))
    assert prompts.WORKSPACE_TEMP_BLOCK.count(render._TEMP_WORKSPACE) == 1
