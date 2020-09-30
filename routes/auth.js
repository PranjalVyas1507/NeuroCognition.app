/**
 * Created by pranjal on 22-09-2020.
 */
const express = require('express')
const passport = require('passport')
const router = express.Router();
//const

/*
 app.get('/auth/google/callback', passport.authenticate('google', { failureRedirect: '/googleerror' }), function(req, res) {
 res.redirect('/google/' + token); // Redirect user with newly assigned token
 });

 app.get('/auth/twitter/callback', passport.authenticate('twitter', { failureRedirect: '/twittererror' }), function(req, res) {
 res.redirect('/twitter/' + token); // Redirect user with newly assigned token
 });
 */

// @desc    Auth with Google
// @route   GET /auth/google
router.get('/google', passport.authenticate('google', {  prompt :'select_account' ,scope: ['profile'] }));

// @desc    Google auth callback
// @route   GET /auth/google/callback
router.get(
    '/google/callback',
    passport.authenticate('google', { failureRedirect: '/' }),
    function(req, res) {
        res.cookie('User',req.user.UserName , { expire : new Date(Date.now() + (1000*60*2) ) });
        res.cookie('Image',req.user.image , { expire : new Date(Date.now() + (1000*60*2) ) });
        //console.log(req.user);
        res.redirect('/')
}
);

// @desc    Github auth callback
// @route   GET /auth/github/callback
router.get('/github',
    passport.authenticate('github', { scope: [ 'profile' ] }));

router.get('/github/callback',
    passport.authenticate('github', { prompt :'select_account' , failureRedirect: '/' }),
    function(req, res) {
        // Successful authentication, redirect home.
        //console.log(req.user);
        res.cookie('User',req.user.UserName , { expire : new Date(Date.now() + (1000*60*2) ) });
        res.cookie('Image',req.user.image , { expire : new Date(Date.now() + (1000*60*2) ) });
        res.redirect('/')
    });

// @desc    Twitter auth callback
// @route   GET /auth/twitter/callback
router.get('/twitter',
    passport.authenticate('twitter'));

router.get('/twitter/callback',
    passport.authenticate('twitter', { failureRedirect: '/' }),
    function(req, res) {
        // Successful authentication, redirect home.
        //console.log(req.user);
        res.cookie('User',req.user.UserName , { expire : new Date(Date.now() + (1000*60*2) ) });
        res.cookie('Image',req.user.image) , { expire : new Date(Date.now() + (1000*60*2) ) };
        res.redirect('/');
    });

// @desc    LinkedIn auth callback
// @route   GET /auth/LinkedIn/callback




// @desc    Logout user
// @route   /auth/logout
router.get('/logout', function (req, res) {
    req.session.destroy(function(err){
        if(err)
        {
            console.log(err) ;
        }
        else
        {
            req.logout();
            res.clearCookie('User');
            res.clearCookie('Image');
            res.redirect('/')
        }
    }) ;

});



module.exports = router ;