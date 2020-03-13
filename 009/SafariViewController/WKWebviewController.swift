//
//  WKWebviewController.swift
//  SafariViewController
//
//  Created by Satyadev on 09/08/17.
//  Copyright © 2017 Satyadev Chauhan. All rights reserved.
//

import UIKit
import WebKit

class WKWebviewController: UIViewController {
    
    var webView: WKWebView
    var url: URL!
    
    @IBOutlet weak var contentView: UIView!
    
    @IBOutlet weak var barView: UIView!
    @IBOutlet weak var urlField: UITextField!
    
    @IBOutlet weak var backButton: UIBarButtonItem!
    @IBOutlet weak var forwardButton: UIBarButtonItem!
    @IBOutlet weak var reloadButton: UIBarButtonItem!
    
    @IBOutlet weak var progressView: UIProgressView!
    
    let webViewKeyPathsToObserve = ["loading", "estimatedProgress"]
    let refreshControl = UIRefreshControl()
    
    required init(coder aDecoder: NSCoder) {
        webView = WKWebView()
        super.init(coder: aDecoder)!
        
        webView.navigationDelegate = self
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        barView.frame = CGRect(x:0, y: 0, width: view.frame.width, height: 30)
        contentView.addSubview(webView)
        
        //Refresh Control
        refreshControl.addTarget(self, action: #selector(WKWebviewController.reload(_:)), for: UIControl.Event.valueChanged)
        webView.scrollView.addSubview(refreshControl)
        
        for keyPath in webViewKeyPathsToObserve {
            webView.addObserver(self, forKeyPath: keyPath, options: .new, context: nil)
        }
        
        if !url.absoluteString.isEmpty {
            loadURL(url)
        }
        
        backButton.isEnabled = false
        forwardButton.isEnabled = false
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    override func viewWillTransition(to size: CGSize, with coordinator: UIViewControllerTransitionCoordinator) {
        barView.frame = CGRect(x:0, y: 0, width: size.width, height: 30)
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        for keyPath in webViewKeyPathsToObserve {
            webView.removeObserver(self, forKeyPath: keyPath, context: nil)
        }
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        
        webView.frame = contentView.frame;
    }
    
    func loadURL(_ url: URL) {
        
        urlField.text = url.absoluteString
        
        let request = URLRequest(url:url)
        webView.load(request)
    }
    
    func updateToolBar() {
        
        forwardButton.isEnabled = webView.canGoForward;
        backButton.isEnabled = webView.canGoBack;
        refreshControl.endRefreshing()
    }
    
    override func observeValue(forKeyPath keyPath: String?, of object: Any?, change: [NSKeyValueChangeKey : Any]?, context: UnsafeMutableRawPointer?) {
        
        guard let keyPath = keyPath else { return }
        
        switch keyPath {
            
        case "loading":
            // If you have back and forward buttons, then here is the best time to enable it
            backButton.isEnabled = webView.canGoBack
            forwardButton.isEnabled = webView.canGoForward
            
        case "estimatedProgress":
            // If you are using a `UIProgressView`, this is how you update the progress
            progressView.isHidden = webView.estimatedProgress == 1
            progressView.setProgress(Float(webView.estimatedProgress), animated: true);
            
        default:
            break
        }
    }
    
    /*
     // MARK: - Navigation
     
     // In a storyboard-based application, you will often want to do a little preparation before navigation
     override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
     // Get the new view controller using segue.destinationViewController.
     // Pass the selected object to the new view controller.
     }
     */
}

extension WKWebviewController: UITextFieldDelegate {
    
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        
        urlField.resignFirstResponder()
        loadURL(URL(string: urlField.text!)!)
        
        return false
    }
}

//MARK: Actions
extension WKWebviewController {
    
    @IBAction func done(_ sender: Any) {
        dismiss(animated: true, completion: nil)
    }
    
    @IBAction func stop(_ sender: Any) {
        
        webView.stopLoading()
        progressView.setProgress(0, animated: true);
        UIApplication.shared.isNetworkActivityIndicatorVisible = false
        
        urlField.selectAll(urlField)
        urlField.becomeFirstResponder()
    }
    
    @IBAction func back(_ sender: UIBarButtonItem) {
        
        webView.goBack()
        urlField.text = webView.url?.absoluteString
    }
    
    @IBAction func forward(_ sender: UIBarButtonItem) {
        
        webView.goForward()
        urlField.text = webView.url?.absoluteString
    }
    
    @IBAction func reload(_ sender: UIBarButtonItem) {
        
        let request = URLRequest(url:webView.url!)
        webView.load(request)
    }
}

//MARK: WKNavigationDelegate

/*! A class conforming to the WKNavigationDelegate protocol can provide
 methods for tracking progress for main frame navigations and for deciding
 policy for main frame and subframe navigations.
 */

extension WKWebviewController: WKNavigationDelegate {
    
    /*! @abstract Decides whether to allow or cancel a navigation.
     @param webView The web view invoking the delegate method.
     @param navigationAction Descriptive information about the action
     triggering the navigation request.
     @param decisionHandler The decision handler to call to allow or cancel the
     navigation. The argument is one of the constants of the enumerated type WKNavigationActionPolicy.
     @discussion If you do not implement this method, the web view will load the request or, if appropriate, forward it to another application.
 
    @available(iOS 8.0, *)
    public func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Swift.Void) {
        
    }*/
    
    
    /*! @abstract Decides whether to allow or cancel a navigation after its
     response is known.
     @param webView The web view invoking the delegate method.
     @param navigationResponse Descriptive information about the navigation
     response.
     @param decisionHandler The decision handler to call to allow or cancel the
     navigation. The argument is one of the constants of the enumerated type WKNavigationResponsePolicy.
     @discussion If you do not implement this method, the web view will allow the response, if the web view can show it.
     */
    @available(iOS 8.0, *)
    public func webView(_ webView: WKWebView, decidePolicyFor navigationResponse: WKNavigationResponse, decisionHandler: @escaping (WKNavigationResponsePolicy) -> Swift.Void) {
        
        decisionHandler(.allow)
    }
    
    
    /*! @abstract Invoked when a main frame navigation starts.
     @param webView The web view invoking the delegate method.
     @param navigation The navigation.
     */
    @available(iOS 8.0, *)
    public func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
        
        UIApplication.shared.isNetworkActivityIndicatorVisible = true
    }
    
    
    /*! @abstract Invoked when a server redirect is received for the main
     frame.
     @param webView The web view invoking the delegate method.
     @param navigation The navigation.
     */
    @available(iOS 8.0, *)
    public func webView(_ webView: WKWebView, didReceiveServerRedirectForProvisionalNavigation navigation: WKNavigation!) {
        
    }
    
    
    /*! @abstract Invoked when an error occurs while starting to load data for
     the main frame.
     @param webView The web view invoking the delegate method.
     @param navigation The navigation.
     @param error The error that occurred.
     */
    @available(iOS 8.0, *)
    public func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        
        let alert = UIAlertController(title: "Error", message: error.localizedDescription, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "Ok", style: .default, handler: nil))
        present(alert, animated: true, completion: nil)
    }
    
    
    /*! @abstract Invoked when content starts arriving for the main frame.
     @param webView The web view invoking the delegate method.
     @param navigation The navigation.
     */
    @available(iOS 8.0, *)
    public func webView(_ webView: WKWebView, didCommit navigation: WKNavigation!) {
        
    }
    
    
    /*! @abstract Invoked when a main frame navigation completes.
     @param webView The web view invoking the delegate method.
     @param navigation The navigation.
     */
    @available(iOS 8.0, *)
    public func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        
        progressView.setProgress(0.0, animated: false)
        UIApplication.shared.isNetworkActivityIndicatorVisible = false
        updateToolBar()
    }
    
    
    /*! @abstract Invoked when an error occurs during a committed main frame
     navigation.
     @param webView The web view invoking the delegate method.
     @param navigation The navigation.
     @param error The error that occurred.
     */
    @available(iOS 8.0, *)
    public func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        
        progressView.setProgress(0.0, animated: false)
        UIApplication.shared.isNetworkActivityIndicatorVisible = false
        updateToolBar()
    }
    
    
    /*! @abstract Invoked when the web view needs to respond to an authentication challenge.
     @param webView The web view that received the authentication challenge.
     @param challenge The authentication challenge.
     @param completionHandler The completion handler you must invoke to respond to the challenge. The
     disposition argument is one of the constants of the enumerated type
     NSURLSessionAuthChallengeDisposition. When disposition is NSURLSessionAuthChallengeUseCredential,
     the credential argument is the credential to use, or nil to indicate continuing without a
     credential.
     @discussion If you do not implement this method, the web view will respond to the authentication challenge with the NSURLSessionAuthChallengeRejectProtectionSpace disposition.
     */
    @available(iOS 8.0, *)
    public func webView(_ webView: WKWebView, didReceive challenge: URLAuthenticationChallenge, completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Swift.Void) {
        
        let cred = URLCredential.init(trust: challenge.protectionSpace.serverTrust!)
        completionHandler(.useCredential, cred)
    }
    
    
    /*! @abstract Invoked when the web view's web content process is terminated.
     @param webView The web view whose underlying web content process was terminated.
     */
    @available(iOS 9.0, *)
    public func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        
    }
}
