"""
serial_comm.py
--------------
Handles USB serial communication with the nodeMCU motor controller.
"""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    logger.warning("pyserial not installed. Serial communication will be simulated.")
    SERIAL_AVAILABLE = False


class SerialComm:
    """Thread-safe serial communication wrapper for nodeMCU."""

    # Commands sent TO the nodeMCU
    CMD_MOVE = "MOVE:{distance:.3f}\n"       # e.g. "MOVE:1.500\n"

    # Responses received FROM the nodeMCU
    RESP_POSITION_REACHED = "POSITION_REACHED"

    def __init__(self, port: str, baud: int = 115200, timeout: float = 60.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._serial: Optional[object] = None
        self._lock = threading.Lock()

    def connect(self) -> bool:
        """Open the serial port."""
        if not SERIAL_AVAILABLE:
            logger.info("[SIMULATED] Serial connected to %s @ %d baud", self.port, self.baud)
            return True
        try:
            self._serial = serial.Serial(
                self.port, self.baud,
                timeout=self.timeout,
                write_timeout=5.0
            )
            logger.info("Serial connected: %s @ %d baud", self.port, self.baud)
            return True
        except serial.SerialException as e:
            logger.error("Failed to open serial port %s: %s", self.port, e)
            return False

    def disconnect(self):
        """Close the serial port."""
        if self._serial and hasattr(self._serial, "close"):
            self._serial.close()
            logger.info("Serial port closed.")

    def send_move_command(self, distance_meters: float) -> bool:
        """
        Send a MOVE command to the nodeMCU.

        Args:
            distance_meters: Distance to move in meters.

        Returns:
            True if sent successfully, False otherwise.
        """
        cmd = self.CMD_MOVE.format(distance=distance_meters)
        with self._lock:
            if not SERIAL_AVAILABLE:
                logger.info("[SIMULATED] Serial TX: %s", cmd.strip())
                return True
            if not self._serial or not self._serial.is_open:
                logger.error("Serial port not open.")
                return False
            try:
                self._serial.write(cmd.encode("utf-8"))
                self._serial.flush()
                logger.info("Serial TX: %s", cmd.strip())
                return True
            except serial.SerialException as e:
                logger.error("Serial write error: %s", e)
                return False

    def wait_for_position_reached(self, timeout: float = 120.0) -> bool:
        """
        Block until the nodeMCU sends POSITION_REACHED or timeout expires.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True if position was reached, False on timeout or error.
        """
        if not SERIAL_AVAILABLE:
            logger.info("[SIMULATED] Waiting for POSITION_REACHED...")
            time.sleep(2)           # Simulate travel time
            logger.info("[SIMULATED] POSITION_REACHED received.")
            return True

        deadline = time.time() + timeout
        with self._lock:
            if not self._serial or not self._serial.is_open:
                logger.error("Serial port not open.")
                return False
            self._serial.timeout = timeout
            while time.time() < deadline:
                try:
                    line = self._serial.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        logger.info("Serial RX: %s", line)
                    if self.RESP_POSITION_REACHED in line:
                        return True
                except serial.SerialException as e:
                    logger.error("Serial read error: %s", e)
                    return False
        logger.warning("Timed out waiting for POSITION_REACHED.")
        return False
