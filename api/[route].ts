// Share the validated Request/Response API and trained model with Netlify.
import handleRequest from '../netlify/functions/api.mts';

export default { fetch: handleRequest };
