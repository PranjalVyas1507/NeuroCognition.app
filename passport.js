/**
 * Created by pranjal on 20-09-2020.
 */
var GOOGLE_CLIENT_ID = '95188641970-b5dpuie1d0kevhgecfogpjfptfank8k9.apps.googleusercontent.com' ;
var GOOGLE_CLIENT_SECRET = 'aaPScCfl7yVlUS1aDRp6RIR0' ;
const GoogleStrategy = require('passport-google-oauth20').Strategy ;
const mongoose = require('mongoose');
require('./Models/DLparams.model');
var User = mongoose.model('User')

GoogleAuthStrategy = function(passport)
{
    passport.use(new GoogleStrategy({
        clientID:     GOOGLE_CLIENT_ID,
        clientSecret: GOOGLE_CLIENT_SECRET,
        callbackURL: '/auth/google/callback'
        //passReqToCallback   : true
    },
        async (accessToken, refreshToken, profile, done) => {
        console.log(profile);
        const newUser = {
            googleId: profile.id,
            UserName: profile.displayName,
            firstname: profile.name.givenName,
            lastname: profile.name.familyName,
            image: profile.photos[0].value
        }

        try {
            let user = await User.findOne({ googleId: profile.id });

        if (user) {
    done(null, user)
} else {
    user = await User.create(newUser)
    done(null, user)
}
} catch (err) {
    console.error(err)
}
}
    )
    );

    passport.serializeUser((user, done) => {
        done(null, user.id)
    })

    passport.deserializeUser((id, done) => {
        User.findById(id, (err, user) => done(err, user))
    })
};

module.exports = GoogleAuthStrategy ;

