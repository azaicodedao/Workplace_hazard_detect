import asyncio
import unittest

from app.services.camera_session_manager import CameraSessionManager


class CameraSessionManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_tokens_are_role_specific(self):
        manager = CameraSessionManager(ttl_seconds=300)
        session = await manager.create()
        self.assertIsNotNone(
            await manager.authenticate(session.session_id, "host", session.host_token)
        )
        self.assertIsNotNone(
            await manager.authenticate(session.session_id, "phone", session.phone_token)
        )
        self.assertIsNone(
            await manager.authenticate(session.session_id, "phone", session.host_token)
        )

    async def test_expired_session_is_removed(self):
        manager = CameraSessionManager(ttl_seconds=0)
        session = await manager.create()
        await asyncio.sleep(0)
        self.assertIsNone(await manager.get(session.session_id))


if __name__ == "__main__":
    unittest.main()
