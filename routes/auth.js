/**
 * Created by pranjal on 22-09-2020.
 */
const express = require('express')
const passport = require('passport')
const router = express.Router()

// @desc    Auth with Google
// @route   GET /auth/google
router.get('/google', passport.authenticate('google', {  prompt :'select_account' ,scope: ['profile'] }))

// @desc    Google auth callback
// @route   GET /auth/google/callback
router.get(
    '/google/callback',
    passport.authenticate('google', { failureRedirect: '/' }),
    function(req, res) {
    res.redirect('/')
}
);

// @desc    Logout user
// @route   /auth/logout
router.get('/logout', function (req, res) {
    req.logout()
    res.redirect('/')
});

module.exports = router ;