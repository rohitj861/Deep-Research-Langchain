"""The password gate on the deployed app."""

import unittest

from tests._harness import app_test, unlock

SECRET = "correct-horse-battery-staple"


class LockedAppTests(unittest.TestCase):
    def test_locked_app_shows_a_password_form(self):
        with app_test(SECRET) as at:
            at.run()
            self.assertEqual([str(e.value) for e in at.exception], [])
            self.assertTrue(at.text_input, "expected a password field")

    def test_locked_app_reveals_nothing_else(self):
        with app_test(SECRET) as at:
            at.run()
            # No provider picker, no depth selector, no chat box until unlocked.
            self.assertEqual(len(at.radio), 0)
            self.assertEqual(len(at.selectbox), 0)
            self.assertEqual(len(at.chat_input), 0)

    def test_wrong_password_is_rejected(self):
        with app_test(SECRET) as at:
            at.run()
            unlock(at, "not-the-password")
            self.assertTrue(any("Incorrect password" in e.value for e in at.error))
            self.assertEqual(len(at.chat_input), 0)

    def test_correct_password_unlocks_the_app(self):
        with app_test(SECRET) as at:
            at.run()
            unlock(at, SECRET)
            self.assertEqual([str(e.value) for e in at.exception], [])
            self.assertTrue(at.chat_input, "expected the app to be unlocked")
            self.assertTrue(at.radio, "expected the provider picker")

    def test_plaintext_is_not_retained_in_session_state(self):
        with app_test(SECRET) as at:
            at.run()
            unlock(at, SECRET)
            self.assertEqual(at.session_state["_password_input"], "")

    def test_a_near_miss_is_still_rejected(self):
        with app_test(SECRET) as at:
            at.run()
            unlock(at, SECRET[:-1])
            self.assertEqual(len(at.chat_input), 0)


class UnprotectedAppTests(unittest.TestCase):
    def test_without_a_password_the_app_runs_but_warns(self):
        with app_test("") as at:
            at.run()
            self.assertTrue(at.chat_input, "app should still work when no password is set")
            self.assertTrue(any("APP_PASSWORD" in w.value for w in at.warning))


if __name__ == "__main__":
    unittest.main()
