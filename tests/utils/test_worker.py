"""Comprehensive tests for Worker thread class."""
import time
import threading
import pytest
from rdoai.utils.worker import Worker


class TestWorkerInitialization:
    """Test Worker initialization."""

    def test_worker_init_with_defaults(self):
        """Test Worker initialization with default maxsize."""
        def dummy_target(item):
            pass

        worker = Worker(name="test-worker", target=dummy_target)
        
        assert worker.name == "test-worker"
        assert worker._target == dummy_target
        assert worker.q.maxsize == 200
        assert isinstance(worker.thread, threading.Thread)
        assert worker.thread.daemon is True

    def test_worker_init_with_custom_maxsize(self):
        """Test Worker initialization with custom maxsize."""
        def dummy_target(item):
            pass

        worker = Worker(name="custom-worker", target=dummy_target, maxsize=50)
        
        assert worker.name == "custom-worker"
        assert worker.q.maxsize == 50

    def test_worker_thread_is_daemon(self):
        """Test that worker thread is created as daemon."""
        def dummy_target(item):
            pass

        worker = Worker(name="daemon-test", target=dummy_target)
        assert worker.thread.daemon is True


class TestWorkerStartStop:
    """Test Worker start and stop methods."""

    def test_worker_start(self):
        """Test that worker thread starts."""
        def dummy_target(item):
            pass

        worker = Worker(name="start-test", target=dummy_target)
        worker.start()
        
        # Give thread a moment to start
        time.sleep(0.05)
        assert worker.thread.is_alive()
        
        # Cleanup
        worker.stop()
        worker.thread.join(timeout=1)

    def test_worker_stop(self):
        """Test that worker thread stops."""
        def dummy_target(item):
            pass

        worker = Worker(name="stop-test", target=dummy_target)
        worker.start()
        time.sleep(0.05)
        
        worker.stop()
        worker.thread.join(timeout=1)
        
        assert not worker.thread.is_alive()

    def test_stop_sets_event(self):
        """Test that stop() sets the stop event."""
        def dummy_target(item):
            pass

        worker = Worker(name="event-test", target=dummy_target)
        assert not worker._stop.is_set()
        
        worker.stop()
        assert worker._stop.is_set()


class TestWorkerSubmit:
    """Test Worker submit method."""

    def test_submit_item(self):
        """Test submitting an item to the worker queue."""
        processed_items = []

        def target(item):
            processed_items.append(item)

        worker = Worker(name="submit-test", target=target)
        worker.start()
        
        worker.submit("test-item")
        time.sleep(0.1)  # Give worker time to process
        
        assert "test-item" in processed_items
        
        worker.stop()
        worker.thread.join(timeout=1)

    def test_submit_multiple_items(self):
        """Test submitting multiple items."""
        processed_items = []

        def target(item):
            processed_items.append(item)

        worker = Worker(name="multi-submit-test", target=target)
        worker.start()
        
        for i in range(5):
            worker.submit(f"item-{i}")
        
        time.sleep(0.2)  # Give worker time to process all items
        
        assert len(processed_items) == 5
        for i in range(5):
            assert f"item-{i}" in processed_items
        
        worker.stop()
        worker.thread.join(timeout=1)

    def test_submit_when_queue_full_drops_item(self):
        """Test that submit drops items when queue is full."""
        processed_items = []
        block_event = threading.Event()

        def blocking_target(item):
            # Block until we signal
            block_event.wait()
            processed_items.append(item)

        # Use very small queue
        worker = Worker(name="full-queue-test", target=blocking_target, maxsize=1)
        worker.start()
        
        # Fill the queue (maxsize=1) - first item goes to thread, not queue
        worker.submit("item-1")
        time.sleep(0.05)  # Let it get picked up by worker thread
        
        # Queue should now have capacity for 1 item
        worker.submit("item-2")  # This fills the queue
        time.sleep(0.01)
        
        # This should be dropped because queue is full (1 in thread, 1 in queue)
        worker.submit("item-3")
        worker.submit("item-4")  # Also dropped
        
        # Unblock and let items process
        block_event.set()
        time.sleep(0.3)
        
        # Should only have processed item-1 and item-2 (item-3 and item-4 were dropped)
        assert len(processed_items) <= 2
        assert "item-1" in processed_items
        # item-2 might or might not have made it depending on timing
        assert "item-3" not in processed_items
        assert "item-4" not in processed_items
        
        worker.stop()
        worker.thread.join(timeout=1)


class TestWorkerExecution:
    """Test Worker task execution."""

    def test_target_function_execution(self):
        """Test that target function is executed with correct argument."""
        received_items = []

        def target(item):
            received_items.append(item)

        worker = Worker(name="exec-test", target=target)
        worker.start()
        
        test_item = {"key": "value"}
        worker.submit(test_item)
        time.sleep(0.1)
        
        assert len(received_items) == 1
        assert received_items[0] == test_item
        
        worker.stop()
        worker.thread.join(timeout=1)

    def test_exception_in_target_does_not_kill_thread(self):
        """Test that exceptions in target function don't kill the worker thread."""
        processed_items = []

        def failing_target(item):
            if item == "bad":
                raise ValueError("Intentional error")
            processed_items.append(item)

        worker = Worker(name="exception-test", target=failing_target)
        worker.start()
        
        worker.submit("good-1")
        worker.submit("bad")  # This will raise an exception
        worker.submit("good-2")
        
        time.sleep(0.2)
        
        # Worker should still be alive and processing after the exception
        assert worker.thread.is_alive()
        assert "good-1" in processed_items
        assert "good-2" in processed_items
        assert len(processed_items) == 2  # "bad" was not added due to exception
        
        worker.stop()
        worker.thread.join(timeout=1)

    def test_none_item_is_skipped(self):
        """Test that None items are skipped by the worker."""
        processed_items = []

        def target(item):
            processed_items.append(item)

        worker = Worker(name="none-test", target=target)
        worker.start()
        
        worker.submit("item-1")
        worker.submit(None)  # Should be skipped
        worker.submit("item-2")
        
        time.sleep(0.2)
        
        # None should not be in processed items
        assert len(processed_items) == 2
        assert None not in processed_items
        assert "item-1" in processed_items
        assert "item-2" in processed_items
        
        worker.stop()
        worker.thread.join(timeout=1)


class TestWorkerIntegration:
    """Integration tests for Worker."""

    def test_worker_processes_items_in_order(self):
        """Test that worker processes items in FIFO order."""
        processed_items = []
        lock = threading.Lock()

        def target(item):
            with lock:
                processed_items.append(item)

        worker = Worker(name="order-test", target=target)
        worker.start()
        
        items = [1, 2, 3, 4, 5]
        for item in items:
            worker.submit(item)
        
        time.sleep(0.2)
        
        assert processed_items == items
        
        worker.stop()
        worker.thread.join(timeout=1)

    def test_worker_with_different_data_types(self):
        """Test worker with various data types."""
        processed_items = []

        def target(item):
            processed_items.append(item)

        worker = Worker(name="types-test", target=target)
        worker.start()
        
        test_items = [
            42,
            "string",
            {"dict": "value"},
            ["list", "item"],
            (1, 2, 3),
        ]
        
        for item in test_items:
            worker.submit(item)
        
        time.sleep(0.2)
        
        assert len(processed_items) == len(test_items)
        for item in test_items:
            assert item in processed_items
        
        worker.stop()
        worker.thread.join(timeout=1)
