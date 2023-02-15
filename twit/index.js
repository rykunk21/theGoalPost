require("dotenv").config({ path: __dirname + "/.env" });

const { twitterClient } = require("./twitterClient.js")

const tweet = async (contents) => {
  try {
    await twitterClient.v2.tweet(contents.body);
  } catch (e) {
    console.log(e)
  }
}

tweet({body: 'Hello World!'});



