// Share the validated Request/Response API and trained model with Netlify.
import handleRequest from '../netlify/functions/api.ts';

export default { fetch: handleRequest };
