from visual_search.db.repository import FeedbackRepository


def test_feedback_and_search_history(tmp_path):
    repository = FeedbackRepository(tmp_path / "events.sqlite3")
    repository.record_feedback("u1", "/images/a.jpg", "like")
    repository.record_feedback("u1", "/images/a.jpg", "dislike")
    assert repository.liked_paths("u1") == []

    repository.record_feedback("u1", "/images/b.jpg", "like")
    assert repository.liked_paths("u1") == ["/images/b.jpg"]

    repository.record_search("u1", "image", "uploaded image", 5)
    history = repository.recent_searches("u1")
    assert len(history) == 1
    assert history[0]["result_count"] == 5


def test_invalid_feedback_action_is_rejected(tmp_path):
    repository = FeedbackRepository(tmp_path / "events.sqlite3")
    try:
        repository.record_feedback("u1", "x.jpg", "maybe")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
