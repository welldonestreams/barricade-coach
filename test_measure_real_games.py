import unittest

import measure_real_games as measure


def row(history, move, build='build-a', side='red'):
    return dict(history=history, top=[[0, move]], build=build,
                position=dict(side=side))


class RealGameMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.game = dict(shareCode='abc123', historyCsv='e2,e8,e3', winner=1,
                         player1Username='steak2222', player2Username='friend',
                         p1Rating=1300, p2Rating=1400)
        self.accounts = {'steak2222'}

    def test_normalizes_only_real_game_identifiers(self):
        self.assertEqual(measure.normalize_game_id('abc123'), 'abc123')
        self.assertEqual(measure.normalize_game_id('/game/abc123'), 'abc123')
        self.assertEqual(measure.normalize_game_id(
            'https://barricade.gg/analysis?game=abc123&ref=x'), 'abc123')
        self.assertIsNone(measure.normalize_game_id('review/abc123'))

    def test_requires_every_recommendation_to_be_followed(self):
        audit = measure.audit_game(self.game,
                                   [row([], 'e2'), row(['e2', 'e8'], 'e3')],
                                   self.accounts, '')
        self.assertTrue(audit['fully_followed'])
        self.assertEqual(audit['coverage'], 1)
        self.assertEqual(audit['follow_rate'], 1)
        self.assertEqual(audit['build'], 'build-a')

    def test_diverged_game_is_not_fully_followed(self):
        audit = measure.audit_game(self.game,
                                   [row([], 'd1'), row(['e2', 'e8'], 'e3')],
                                   self.accounts, '')
        self.assertEqual(audit['status'], 'diverged')
        self.assertFalse(audit['fully_followed'])
        self.assertEqual(audit['divergences'][0]['ply'], 1)

    def test_partial_and_mixed_build_games_are_excluded(self):
        partial = measure.audit_game(self.game, [row([], 'e2')], self.accounts, '')
        self.assertEqual(partial['status'], 'partial')
        mixed = measure.audit_game(
            self.game, [row([], 'e2'), row(['e2', 'e8'], 'e3', 'build-b')],
            self.accounts, '')
        self.assertEqual(mixed['status'], 'mixed_builds')
        self.assertFalse(mixed['fully_followed'])


if __name__ == '__main__':
    unittest.main()
