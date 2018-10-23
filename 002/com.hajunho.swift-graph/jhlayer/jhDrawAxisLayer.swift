//
//  jhDrawAxisLayer.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 23..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

class jhDrawAxisLayer : CALayer {
    
    private let panelID: Int
    
    private var mContext: CGContext?
    
    /// Axes
    private var mUnitOfHorizontalAxes : CGFloat = 100
    private var mcountOfHorizontalAxes : Int = 3
    
    internal var mLineWidth : CGFloat = 1
    internal var mColor : CGColor = UIColor.blue.cgColor
//        UIColor(red: 229, green: 229, blue: 229, alpha: 1.0).cgColor
    
    var axisDistance, mVerticalRatioToDraw_view, mMargin, mPanelWidth, mPanelHeight, mFixedPanelWidth, mFixedPanelHeight : CGFloat
    
    init(_ axisD: CGFloat, _ vRatio: CGFloat, _ margin: CGFloat, _ w: CGFloat, _ h: CGFloat, _ fw: CGFloat, _ fh: CGFloat, layer: Any, panelID: Int) {
        self.mContext = UIGraphicsGetCurrentContext()
        self.axisDistance = axisD
        self.mVerticalRatioToDraw_view = vRatio
        self.mMargin = margin
        self.mPanelWidth = w
        self.mPanelHeight = h
        self.mFixedPanelWidth = fw
        self.mFixedPanelHeight = fh
        
        self.panelID = panelID
        
        super.init(layer: layer)
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func draw(in ctx: CGContext) {
        
        mContext = ctx
        
        var xlocation : CGFloat = 0
        
        for x in 1..<jhDataCenter.mCountOfaxes_view+1 {
            
            xlocation = CGFloat(x) * axisDistance + mMargin
            
            drawLine(xlocation, mMargin, xlocation, jhDraw.maxR-mMargin)
            
            //TODO: LABEL
            //            self.contents = (drawText(str: String(x), x: xlocation-10, y: jhDraw.maxR-mMargin, width: 10, height: 10)).cgImage
            
            self.addSublayer(drawText(str: String(x), x: xlocation-10, y: jhDraw.maxR-mMargin, width: 10, height: 10))
        }
        
        for x in 1..<mcountOfHorizontalAxes+1 {
            let fx = CGFloat(x)*mUnitOfHorizontalAxes*mVerticalRatioToDraw_view + mMargin
            drawLine(mMargin, fx, jhDraw.maxR-mMargin, fx)
            
            //TODO: LABEL
            
            self.addSublayer(drawText(str: String(x), x: 100, y: fx, width: 10, height: 10))
        }
        
        //TODO: warning guide line. There's a bug.
        drawLineWithColor(mMargin, 20*mUnitOfHorizontalAxes, jhDraw.maxR-mMargin, 20*mUnitOfHorizontalAxes, lineWidth: 2, color: UIColor(red: 254, green: 191, blue: 4, alpha: 0.5).cgColor)
        drawLineWithColor(mMargin, 60*mUnitOfHorizontalAxes, jhDraw.maxR-mMargin, 60*mUnitOfHorizontalAxes, lineWidth: 2, color: UIColor(red: 251, green: 83, blue: 96, alpha: 0.5).cgColor)
    }
    
    private func drawLine(_ x1 : CGFloat, _ y1 : CGFloat, _ x2 : CGFloat, _ y2 : CGFloat) {
        jhDraw.worldLine(context: mContext, getX(x1)!, getY(y1)!, getX(x2)!, getY(y2)!, mLineWidth, mColor)
    }
    
    private func drawLineWithColor(_ x1 : CGFloat, _ y1 : CGFloat, _ x2 : CGFloat, _ y2 : CGFloat, lineWidth : CGFloat, color : CGColor) {
        jhDraw.worldLine(context: mContext, getX(x1)!, getY(y1)!, getX(x2)!, getY(y2)!, lineWidth, color)
    }
    
    private func getX(_ x: CGFloat) -> CGFloat? {
        var retX : CGFloat? = nil
        retX = x * mPanelWidth / mFixedPanelWidth
        return retX
    }
    
    private func getY(_ y: CGFloat) -> CGFloat? {
        var retY : CGFloat? = nil
        retY = y * mPanelHeight / mFixedPanelHeight
        return retY
    }
    
    
    private func drawText(str : String, x : CGFloat, y : CGFloat, width : CGFloat, height : CGFloat) -> CALayer {
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height))
        let img = renderer.image { ctx in
            let paragraphStyle = NSMutableParagraphStyle()
            paragraphStyle.alignment = .center
//            let attrs = [NSAttributedString.Key.font: UIFont(name: "".font1(), size: width/2)!, NSAttributedString.Key.paragraphStyle: paragraphStyle]
            let string = str
//            string.draw(with: CGRect(x: 0, y: 0, width: width, height: 10), options: .usesLineFragmentOrigin, attributes: attrs, context: nil)
        }
        //        let imageView : UIImageView = UIImageView(frame: CGRect(x: getX(x)!, y: getY(y)!, width: width, height: height))
        //
        //        imageView.image = img
        
        let tLayer = CALayer()
        let tImg = img.cgImage
        tLayer.frame = CGRect(x: getX(x)!, y: getY(y)!, width: width, height: height)
        tLayer.contents = tImg
        
        return tLayer
    }
}
