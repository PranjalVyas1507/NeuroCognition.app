/**
 * Created by pranjal on 21-09-2020.
 */


const express = require('express');
const router = express.Router();
const Ensure = require('../middleware/auth');



router.get('/',function(req , res){
    res.render('login')

});




module.exports = router;
