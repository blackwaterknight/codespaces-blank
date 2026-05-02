module.exports = async function (context, req) {

    const fetch = require("node-fetch");

    const AZURE_ENDPOINT = "https://HealSolution.services.ai.azure.com/api/projects/HealSolution-Demo/applications/PatientCareCompanion/protocols/openai/responses?api-version=2025-11-15-preview";

    const API_KEY = "3p1AqBXJ9IpQyBrD8bRlkskAOFbCu7AfaqLDQAniHGJBlNNYym75JQQJ99CDACYeBjFXJ3w3AAAAACOGCKC2";

    try {
        const response = await fetch(AZURE_ENDPOINT, {
            method: "POST",
            headers: {
                "api-key": API_KEY,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                input: req.body.message
            })
        });

        const data = await response.json();

        context.res = {
            status: 200,
            body: data
        };

    } catch (error) {
        context.res = {
            status: 500,
            body: { error: error.toString() }
        };
    }
};
