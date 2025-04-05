const axios = require('axios');
const qs = require('querystring');

// 🎉 Token Data: {
//     access_token: 'BQCeRJja1n_56F6x0ocw_wm2Z3jY9c3Wm6tTb5BCk_f7REs2YkFIVaF1mYc0SosS6Q0De7r24-vMLx1T5pTpbVP9ht3QBDs6HcOLjba8IEDqCb5mfE0kCTq5QjOa2hwhOrylZRy6NGxUFnQ3bsOhVI-4_yxLrgklulShMYfJ9OmAD822dsgRAQlSijP2jGcvg4lxje8VnylsazncLIALq0MgzZ8de2SZO68Sjw4ppB6m4eMpdwDuiKIQGQ',
//     token_type: 'Bearer',
//     expires_in: 3600,
//     refresh_token: 'AQCgjYIMSZIGEeSKavbmH9j3PMGC_Vxu7jgBeUSEEf8_1HPwSAbSLvwzMM7ig9OHUMQ7MFZ1odO2QLSpXCh2fLJ64bneEb2vVO-JusumiWqg7cfmNbD6sDM_bzbo_Sx14nA',
//     scope: 'user-read-email user-read-private'

async function getAccessToken(code) {
  const clientId = 'dd7a61b575aa478688a7335d9d01439d';
  const clientSecret = 'c0a9505747074c2fade60dff376b4b59';
  const redirectUri = 'http://localhost:8888/callback';

  const tokenEndpoint = 'https://accounts.spotify.com/api/token';

  const response = await axios.post(tokenEndpoint, qs.stringify({
    grant_type: 'authorization_code',
    code,
    redirect_uri: redirectUri
  }), {
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      'Authorization': 'Basic ' + Buffer.from(`${clientId}:${clientSecret}`).toString('base64')
    }
  });

  return response.data; // Contains access_token, refresh_token, etc.
}

const express = require('express');
const app = express();
const port = 8888;

app.get('/callback', async (req, res) => {
  const code = req.query.code;

  if (!code) {
    return res.status(400).send('Missing code in query');
  }

  try {
    const tokenData = await getAccessToken(code);
    console.log('🎉 Token Data:', tokenData);

    // You could save tokenData.access_token & refresh_token here
    res.send(`Access Token: ${tokenData.access_token}`);
  } catch (err) {
    console.error('Token exchange failed:', err.response?.data || err.message);
    res.status(500).send('Failed to exchange code for token');
  }
});

app.listen(port, () => {
  console.log(`🚀 Server running at http://localhost:${port}`);
});
